import dnois
from dnois.optics import rt
import torch

import conf
import utils


def make_17_points(z, r):
    angles = torch.arange(8) * torch.pi / 4
    x = angles.cos() * r
    y = angles.sin() * r
    points = torch.stack([
        torch.tensor([0, 0, z]),
        *torch.stack([x / 2, y / 2, torch.full_like(angles, z)], -1).unbind(),
        *torch.stack([x, y, torch.full_like(angles, z)], -1).unbind(),
    ])  # (N_s,3)
    return points


def make_more_points(z, r, n=11):
    apt = rt.CircularAperture(r)
    x, y = apt.sample_unipolar(n, 8)
    z = torch.full_like(x, z)
    points = torch.stack([x, y, z], -1)
    return points


def trace(optics, sampling_size, points):
    entry_points = optics.first.sample('rect', sampling_size)  # (spp, 3)
    n_spp = entry_points.size(0)
    intensity = 1 / n_spp
    wl = optics.new_tensor(conf.wl).reshape(-1).unsqueeze(-1)  # (N_wl,1)
    ray_in = rt.BatchedRay(points, entry_points - points, wl, init_opl=0, init_intensity=intensity)
    ray_out = optics.trace_ray(ray_in)  # (N_s,N_wl,spp)
    return ray_in, ray_out


def make_interpolators(ray_out, n_s, n_order):
    position = ray_out.o.reshape(n_s, n_order, -1, 3)[..., :2]  # (N_s, N_order, spp, 2)
    value = torch.stack([
        ray_out.intensity.sqrt(), ray_out.opl * dnois.k(ray_out.wl)
    ], -1).reshape(n_s, n_order, -1, 2)  # (N_s, N_order, spp, 2)
    valid = ray_out.valid.reshape(n_s, n_order, -1)  # (N_s, N_order, spp)

    position = [
        [p[v] for p, v in zip(p1.unbind(), v1.unbind()) if v.any()]
        for p1, v1 in zip(position.unbind(), valid.unbind())
    ]  # (N_s, N_order, spp', 2)
    value = [
        [val[v] for val, v in zip(val1.unbind(), v1.unbind()) if v.any()]
        for val1, v1 in zip(value.unbind(), valid.unbind())
    ]  # (N_s, N_order, spp', 2)
    interpolators = [
        [
            utils.none_interpolator(p.detach().cpu().numpy(), v.detach().cpu().numpy(), 0.)
            for p, v in zip(p1, v1)
        ]
        for p1, v1 in zip(position, value)
    ]

    return interpolators


def interpolate(interpolators, x, y):
    interpolated = [
        [utils.none_interpolate(intp, x, y) for intp in intp1]
        for intp1 in interpolators
    ]  # (H,W,2)
    # for i in range(len(interpolated)):
    #     print(f'Point {i}')
    #     for j in range(len(interpolated[i])):
    #         if interpolated[i][j] is None:
    #             continue
    #         # print(interpolated[i][j][..., 0].mean(), interpolated[i][j][..., 0].std())
    #         # print(interpolated[i][j][..., 1].min(), interpolated[i][j][..., 1].max())
    interpolated_field = [
        [field[..., 0] * dnois.expi(field[..., 1]) for field in field_single_point if field is not None]
        for field_single_point in interpolated
    ]
    interpolated_field = sum([sum(intp, torch.tensor(0.)).abs().square() for intp in interpolated_field])  # (H,W)
    return interpolated_field


def irradiance_from_point(optics, point, sampling_size: int):
    _, ray_out = trace(optics, sampling_size, point)

    grating = optics.surfaces[1]
    interpolators = make_interpolators(ray_out, point.size(0), grating.n_order)

    sensor = optics.sensor
    y, x = dnois.utils.grid(sensor.resolution, sensor.pixel_size, symmetric=True, device='cpu')
    y = -y
    # x=x.contiguous()-5e-3
    interpolated_field = interpolate(interpolators, x, y)

    return dnois.torch.abs2(interpolated_field)

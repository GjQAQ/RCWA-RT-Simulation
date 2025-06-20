import functools

import dnois
import numpy as np
import scipy
import torch
from dnois.optics import rt


def print_order_wise(x, print_dim):
    dims = tuple(range(x.ndim - print_dim))
    print(f'{x.mean(dims)=}, {torch.allclose(x.std(dims), torch.zeros_like(x.std(dims)))}')


class CustomizedGrating(rt.Planar):
    def __init__(
        self,
        transmittance: np.ndarray,  # (N_theta,N_phi,N_order)
        phase_shift: np.ndarray,  # (N_theta,N_phi,N_order)
        direction: np.ndarray,  # (N_theta,N_phi,N_order,3)
        theta: np.ndarray,  # (N_theta,)
        phi: np.ndarray,  # (N_phi,)
        material: dnois.mt.Material | str = 'vacuum',
        aperture: rt.Aperture | float = None,
        expand_dim: int = -1,
        *,
        d=None
    ):
        super().__init__(material, aperture, False, d=d)
        self.expand_dim = expand_dim
        self.n_order = transmittance.shape[-1]

        theta, phi = np.meshgrid(theta, phi, indexing='ij')
        points = np.stack([theta, phi], axis=-1).reshape(-1, 2)  # (N_theta*N_phi,2)
        self.t_intp = scipy.interpolate.LinearNDInterpolator(points, transmittance.reshape(-1, self.n_order))
        self.p_intp = scipy.interpolate.LinearNDInterpolator(points, phase_shift.reshape(-1, self.n_order))
        self.d_intp = scipy.interpolate.LinearNDInterpolator(points, direction.reshape(-1, self.n_order, 3))

    def refract(self, ray: rt.BatchedRay, forward: bool = True) -> rt.BatchedRay:
        if not forward:
            raise NotImplementedError(f'{self.__class__.__name__} cannot be used in backward ray tracing')

        local_ray = self.ctx.g2l_ray(ray).broadcast()
        d_local = local_ray.d

        theta = d_local[..., 2].acos()
        sin_theta_abs = torch.sqrt(d_local[..., :2].square().sum(-1))
        phi = dnois.utils.Conditional(
            lambda x: x < 1e-5,
            lambda _: torch.zeros_like(theta),
            lambda _: torch.remainder(torch.atan2(d_local[..., 1], -d_local[..., 0]) + 2 * np.pi, 2 * np.pi),
        )(sin_theta_abs)
        # print(f'{theta.mean().item()=}, {theta.std().item()=}, {phi.mean().item()=}, {phi.std().item()=}')
        # print(f'{theta=}, {phi=}')
        intp_points = torch.stack([theta, phi], dim=-1).detach().cpu().numpy()  # (..., 2)

        d = self.d_intp(intp_points)  # (..., N_order, 3)
        d = torch.from_numpy(d).to(dtype=self.dtype, device=self.device)
        # print_order_wise(d, 2)
        d = d.unbind(-2)
        d = torch.cat(d, dim=self.expand_dim if self.expand_dim >= 0 else self.expand_dim - 1)
        valid = torch.sum(d.square(), -1) > 0.9
        new_ray = local_ray.expand_dim(self.expand_dim, self.n_order)
        original_d = new_ray.d
        new_ray.d = d

        if new_ray.recording_intensity:
            t = self.t_intp(intp_points)  # (..., N_order)
            t = torch.from_numpy(t).to(dtype=self.dtype, device=self.device)
            # print_order_wise(t, 1)
            t = t.unbind(-1)
            t = torch.cat(t, dim=self.expand_dim)
            new_ray.intensity = new_ray.intensity * t

        if new_ray.coherent:
            p = self.p_intp(intp_points)  # (..., N_order)
            p = torch.from_numpy(p).to(dtype=self.dtype, device=self.device)
            # print_order_wise(p, 1)
            p = p.unbind(-1)
            p = torch.cat(p, dim=self.expand_dim)
            k1 = dnois.k(new_ray.wl).unsqueeze(-1) * original_d
            k2 = dnois.k(new_ray.wl).unsqueeze(-1) * new_ray.d
            p_shift = torch.sum((k2 - k1) * new_ray.o, -1) + p
            if new_ray.recording_phase:
                new_ray.phase = new_ray.phase + p_shift
            if new_ray.recording_opl:
                new_ray.opl = new_ray.opl + p_shift / dnois.k(new_ray.wl)

        new_ray = self.ctx.l2g_ray(new_ray)
        new_ray = new_ray.update_valid(valid)
        return new_ray

    def reflect(self, ray: rt.BatchedRay) -> rt.BatchedRay:
        raise NotImplementedError()

    def backward_valid(self, valid):
        split_valid = valid.chunk(self.n_order, self.expand_dim)
        valid = functools.reduce(torch.logical_or, split_valid)
        return valid

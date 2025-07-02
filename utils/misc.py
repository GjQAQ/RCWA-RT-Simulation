import dnois
from dnois.optics import rt
import numpy as np
import scipy
import torch

import conf


def overwrite_dnois_ray_march():
    def _march_(self, t, n=None):
        if not torch.is_tensor(t):
            t = self.new_tensor(t)
        self.o = self._ts['o'] + self.d * t.unsqueeze(-1)
        _opl, _ph = self.recording_opl, self.recording_phase
        # if _opl or _ph:
        #     opl = t if n is None else n * t
        #     if _opl:
        #         self.opl = self.opl + opl
        #     if _ph:
        #         #         self.phase = self.phase + base.physics.k(self.wl) * opl
        return self

    rt.BatchedRay.march_ = _march_


def thetas(n=None):
    if n is None:
        n = conf.theta_n
    return np.deg2rad(np.linspace(*conf.theta_range, n))


def phis(n=None):
    if n is None:
        n = conf.phi_n
    return np.deg2rad(np.linspace(*conf.phi_range, n))


def d_in_air(d):
    n = dnois.mt.refractive_index(conf.wl, conf.substrate_material)
    d = torch.from_numpy(d)
    d = dnois.refract(d, d.new_tensor([0, 0, 1]), n)
    return d.cpu().numpy()


def none_interpolator(p, v, fill_value):
    try:
        return scipy.interpolate.LinearNDInterpolator(p, v, fill_value=fill_value)
    except Exception as e:
        print(f'ERROR: {e.__class__.__name__}')
        return None


def none_interpolate(interpolator, *args):
    if interpolator is None:
        return None
    return torch.from_numpy(interpolator(*args))

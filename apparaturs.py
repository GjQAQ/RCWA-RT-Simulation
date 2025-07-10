from dnois.optics import rt
import numpy as np

import grating
import utils


def layout(transmittance, phase_shift, out_d):
    theta = np.deg2rad(np.linspace(0, 55, 12))
    sq = rt.CoaxialSurfaceSequence([
        rt.ThinLens(5e-3, 'vacuum', 10.325e-3, d=5e-3),
        grating.CustomizedGrating(
            transmittance, phase_shift, out_d, theta, utils.phis(), 'vacuum', 5e-3, -1, d=6.25e-3
        ),
        rt.ThinLens(6.25e-3, 'vacuum', 12.906e-3, d=6.25e-3),
        rt.CircularStop(12.906e-3, d=62.5e-3),
        rt.ThinLens(62.5e-3, 'vacuum', 129.06e-3, d=62.5e-3),
    ])
    sq.freeze()
    return sq


def layout_reflective(transmittance, phase_shift, out_d):
    theta = np.deg2rad(np.linspace(0, 55, 12))
    sq = rt.CoaxialSurfaceSequence([
        rt.ThinLens(6.25e-3, 'vacuum', 12.906e-3, d=6.25e-3),
        grating.CustomizedGrating(
            transmittance, phase_shift, out_d, theta, utils.phis(), 'vacuum', 8e-3, -1, d=-6.25e-3
        ),
        rt.ThinLens(6.25e-3, 'vacuum', 12.906e-3, d=-6.25e-3),
        rt.CircularStop(12.906e-3, d=-62.5e-3),
        rt.ThinLens(62.5e-3, 'vacuum', 129.06e-3, d=-62.5e-3),
    ])
    sq[1].reflective = True
    sq.freeze()
    return sq


def layout_fixed(transmittance, phase_shift, out_d):
    sq = rt.CoaxialSurfaceSequence([
        rt.ThinLens(5e-3, 'vacuum', 10.325e-3, d=5e-3),
        grating.FixedGrating(transmittance, phase_shift, out_d, 'vacuum', 5e-3, d=6.25e-3),
        rt.ThinLens(6.25e-3, 'vacuum', 12.906e-3, d=6.25e-3),
        rt.CircularStop(12.906e-3, d=62.5e-3),
        rt.ThinLens(62.5e-3, 'vacuum', 129.06e-3, d=62.5e-3),
    ])
    sq.freeze()
    return sq

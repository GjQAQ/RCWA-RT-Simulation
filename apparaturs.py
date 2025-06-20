import math

from dnois.optics import rt
import numpy as np

import grating
import utils


def layout1(transmittance, phase_shift, out_d):
    sq = rt.CoaxialSurfaceSequence([
        rt.ThinLens(5e-3, 'vacuum', 0.8749e-3, d=5e-3),
        grating.CustomizedGrating(
            transmittance, phase_shift, out_d, utils.thetas(), utils.phis(), 'vacuum', 5e-3 * 2, -1, d=5e-3
        ),
        rt.ThinLens(6.25e-3, 'vacuum', 5e-3 * 2, d=6.25e-3),
        rt.CircularStop(25.4e-3 * 2, d=62.5e-3),
        rt.ThinLens(62.5e-3, 'vacuum', 25e-3 * 2, d=0.18725),
    ])
    sq.freeze()
    return sq


def layout2(transmittance, phase_shift, out_d):  # RCWA_grating_Kohler_0530_more.zmx
    sq = rt.CoaxialSurfaceSequence([
        rt.ThinLens(5e-3, 'vacuum', 0.8749e-3, d=5e-3),  # corresponding to a divergence angle of 5°
        grating.CustomizedGrating(
            transmittance, phase_shift, out_d, utils.thetas(), utils.phis(), 'vacuum', 5e-3 * 2, -1, d=5e-3
        ),
        rt.ThinLens(6.25e-3, 'vacuum', 12.906e-3 * 2, d=6.25e-3),
        rt.CircularStop(1.6e-3 * 2, d=62.5e-3),
        rt.ThinLens(62.5e-3, 'vacuum', 12.5e-3 * 2, d=62.5e-3),
    ])
    sq.freeze()
    return sq


def layout3(transmittance, phase_shift, out_d):
    sq = rt.CoaxialSurfaceSequence([
        rt.ThinLens(5e-3, 'vacuum', 0.8749e-3, d=5e-3),
        grating.CustomizedGrating(
            transmittance, phase_shift, out_d, utils.thetas(11), utils.phis(), 'vacuum', 5e-3 * 2, -1, d=5e-3
        ),
        rt.ThinLens(6.25e-3, 'vacuum', 12.906e-3 * 2, d=6.25e-3),
        rt.CircularStop(10e-3 * 2, d=62.5e-3),
        rt.ThinLens(62.5e-3, 'vacuum', 12.5e-3 * 2, d=62.5e-3),
    ])
    sq.freeze()
    return sq


def layout4(transmittance, phase_shift, out_d):
    sq = layout2(transmittance, phase_shift, out_d)
    sq[1].ctx.theta = math.radians(3)
    sq.freeze()
    return sq


def layout0610(transmittance, phase_shift, out_d):
    theta = np.deg2rad(np.linspace(0, 55, 12))
    sq = rt.CoaxialSurfaceSequence([
        rt.ThinLens(5e-3, 'vacuum', 10.325e-3 * 2, d=5e-3),  # corresponding to a divergence angle of 5°
        grating.CustomizedGrating(
            transmittance, phase_shift, out_d, theta, utils.phis(), 'vacuum', 50e-3 * 2, -1, d=6.25e-3
        ),
        rt.ThinLens(6.25e-3, 'vacuum', 12.906e-3 * 2, d=6.25e-3),
        rt.CircularStop(12.906e-3 * 2, d=62.5e-3),
        rt.ThinLens(62.5e-3, 'vacuum', 129.06e-3 * 2, d=62.5e-3),
    ])
    sq.freeze()
    return sq

def layout0620(transmittance, phase_shift, out_d):
    theta = np.deg2rad(np.linspace(0, 55, 12))
    sq = rt.CoaxialSurfaceSequence([
        rt.ThinLens(6.25e-3, 'vacuum', 12.906e-3 * 2, d=6.25e-3),  # corresponding to a divergence angle of 5°
        grating.CustomizedGrating(
            transmittance, phase_shift, out_d, theta, utils.phis(), 'vacuum', 8e-3 * 2, -1, d=-12.5e-3
        ),
        rt.CircularStop(12.906e-3 * 2, d=-62.5e-3),
        rt.ThinLens(62.5e-3, 'vacuum', 129.06e-3 * 2, d=-62.5e-3),
    ])
    sq.freeze()
    return sq

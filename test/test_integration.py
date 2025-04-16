"""Test integration routines for fiber orientation density functions"""

from math import pi

import numpy.testing as npt

from waffleiron.material import integrate_sph2_oct, integrate_sph2_gkt


def test_surface_area_half_sphere():
    """Integration of dA over the unit sphere should return 4π

    The surface area of the unit half-sphere is 2π.
    """
    sa_expected = 2 * pi
    # Hard-coded weights table (less accurate)
    sa = integrate_sph2_oct(0, lambda x: 1)
    npt.assert_almost_equal(sa, sa_expected, decimal=6)
    # GKT scheme (more accurate)
    sa = integrate_sph2_gkt(0, lambda x: 1, o_φ=3, n_θ=15)
    npt.assert_almost_equal(sa, sa_expected, decimal=14)

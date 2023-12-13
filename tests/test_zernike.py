"""Tests for zernike."""
import numpy as np

from ..faster_zernike import (
    ZerothDerivative,
    FirstDerivative,
    SecondDerivative,
    ThirdDerivative,
    FourthDerivative,
)


def test_zernike_radial_prata():
    """Test zernike."""
    # calculate these modes
    l = np.arange(0, 8)
    # radii on unit disc
    r = np.linspace(0.1, 1, 20)

    specific_l = 2
    specific_m = 0

    f = FourthDerivative()
    # calculates all polynomials since it's fourth derivative
    f.zernike_radial_prata(r, l)

    # zeroth derivative
    x = ZerothDerivative().zernike_radial_analytical(r)
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.R[0][specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

    # first derivative
    x = FirstDerivative().zernike_radial_analytical(r)
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.R[1][specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

    # second derivative
    x = SecondDerivative().zernike_radial_analytical(r)
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.R[2][specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

    # third derivative
    x = ThirdDerivative().zernike_radial_analytical(r)
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.R[3][specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

    # fourth derivative
    x = FourthDerivative().zernike_radial_analytical(r)
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.R[4][specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

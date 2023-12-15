"""Tests for zernike."""
import numpy as np

from ..zernike_prata import (
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

    f = FourthDerivative(r, l)
    # calculates all polynomials since it's fourth derivative
    f.zernike_radial_prata()

    # zeroth derivative
    x = ZerothDerivative(r, l).zernike_radial_analytical()
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.get_zeroth_derivative()[specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

    # first derivative
    x = FirstDerivative(r, l).zernike_radial_analytical()
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.get_first_derivative()[specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

    # second derivative
    x = SecondDerivative(r, l).zernike_radial_analytical()
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.get_second_derivative()[specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

    # third derivative
    x = ThirdDerivative(r, l).zernike_radial_analytical()
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.get_third_derivative()[specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

    # fourth derivative
    x = FourthDerivative(r, l).zernike_radial_analytical()
    zernike_analytical = x[str(specific_l) + str(specific_m)]
    zernike_computed = f.get_fourth_derivative()[specific_l, specific_m]
    np.testing.assert_allclose(zernike_analytical, zernike_computed, rtol=1e-3)

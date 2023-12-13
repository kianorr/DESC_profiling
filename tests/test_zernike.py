"""Tests for zernike."""
import numpy as np

from ..faster_zernike import zernike_radial_singh, zernike_direct_eval


def test_zernike_eval():
    """Test zernike."""
    # calculate these modes
    l = np.arange(0, 8)
    # radii on unit disc
    r = np.linspace(0, 1, 20)

    specific_l = 3
    specific_m = 1
    zernike_prata = zernike_radial_singh(r, l)[specific_l, specific_m]
    zernike_direct = zernike_direct_eval(r)[str(specific_l) + str(specific_m)]
    np.testing.assert_allclose(zernike_prata, zernike_direct, rtol=1)

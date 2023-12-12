"""Tests for zernike."""
import numpy as np
from faster_zernike import zernike_radial_singh, zernike_direct_eval


def test_zernike_eval():
    """Test zernike."""
    L = 8
    M = 8
    # TODO: add r without desc
    grid = LinearGrid(L=L, M=M)
    r = grid.nodes[:, 0]

    _l = np.arange(L) + 1
    zernike_prata = zernike_radial_singh(r, _l)[7, 5]
    zernike_direct = zernike_direct_eval(r, _l, m)

    assert zernike_prata == zernike_direct

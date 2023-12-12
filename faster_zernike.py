"""zernike algorithms."""
import numpy as np
from desc.grid import LinearGrid


def zernike_direct_eval(r, n, m):
    """Directly evaluates zernike."""
    factorial = np.math.factorial
    R_nm = np.zeros((n - m) / 2)
    for k in range((n - m) / 2):
        R_nmk = (
            (-1) ** ((n - k) / 2)
            * factorial((n + k) / 2)
            / factorial((n - k) / 2)
            * factorial((k + abs(m)) / 2)
            * factorial((k - abs(m)) / 2)
        )

        R_nm += R_nmk * r**k
    return R_nm


def zernike_radial_singh(r, n):
    """Radial part of zernike polynomials. Modified Prata's algorithm."""
    R = np.full((len(n), len(n), len(r)), np.nan)

    n_max = np.max(n)
    for n_i in range(n_max):
        R[n_i, n_i] = r**n_i

    for n_i in range(2, n_max + 2, 2):
        for m_i in range(0, n_max - n_i):
            K_1 = 2 * n_i / (m_i + n_i)
            K_2 = 1 - K_1

            term_1 = r * K_1 * R[n_i + m_i - 1, abs(m_i - 1)]
            term_2 = K_2 * R[m_i + n_i - 2, m_i]

            R[n_i + m_i, m_i] = term_1 + term_2

    return R


L = 8
M = 8
# TODO: add r without desc
grid = LinearGrid(L=L, M=M)
r = grid.nodes[:, 0]

_l = np.arange(L) + 1
print(zernike_radial_singh(r, _l)[7, 5])

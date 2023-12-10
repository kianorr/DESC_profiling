from desc.grid import LinearGrid
import numpy as np

def zernike_radial_direct(r, n):
    """Radial part of zernike polynomials. Modified Prata's algorithm."""
    R = np.zeros((len(n), len(n), len(r)))  # R_nmk

    M = np.max(n)
    for n_i in range(M):

        R[n_i, n_i] = r ** n_i
        print(f"{R[n_i, n_i]=}")

        for m_i in range(M, 1, -2):
            K_1 = 2 * n_i / (m_i + n_i)
            K_2 = 1 - K_1

            R[n_i, m_i] = r * K_1 * R[n_i - 1, m_i - 1] + K_2 * R[n_i - 2, m_i]

        # covering the case where n is even
        if n_i % 2 == 0:
            R[n_i, 0] = 2 * r * R[n_i - 1, 1] - R[n_i - 2, 0]
    
    return R

L = 8
# TODO: add r without desc
grid = LinearGrid(L=L)
r = grid.nodes[:, 0]

l = np.arange(L)
print(zernike_radial_direct(r, l)[-1])
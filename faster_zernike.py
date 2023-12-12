"""zernike algorithms."""
import numpy as np
import matplotlib.pyplot as plt


def zernike_direct_eval(r):
    """Directly evaluates zernike.

    From https://en.wikipedia.org/wiki/Zernike_polynomials#Radial_polynomials
    """
    R = {}
    R["42"] = 4 * r**4 - 3 * r**2
    R["31"] = 3 * r**3 - 2 * r
    return R


def zernike_radial_singh(r, n):
    """Radial part of zernike polynomials. Modified Prata's algorithm."""
    R = np.ones((len(n), len(n), len(r))) * np.nan

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


def plot_zernike(r, l, m):
    """Plot radial zernike with respect to radii."""
    algo = zernike_radial_singh(r, l)[l, m]
    direct = zernike_direct_eval(r)
    plt.plot(r, algo)
    plt.plot(r, direct)
    plt.show()


l = np.arange(0, 8)
r = np.linspace(0, 1, 20)
algo = zernike_radial_singh(r, l)[3, 1]
direct = zernike_direct_eval(r)
plot_zernike(r, 4, 2)

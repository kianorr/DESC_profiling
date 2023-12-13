"""zernike algorithms."""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def zernike_direct_eval(r):
    """Directly evaluates zernike.

    From https://en.wikipedia.org/wiki/Zernike_polynomials#Radial_polynomials
    """
    R = {}
    R["42"] = 4 * r**4 - 3 * r**2
    R["31"] = 3 * r**3 - 2 * r
    R["20"] = 2 * r ** 2 - 1
    R["40"] = 6 * r ** 4 - 6 * r **2 + 1
    return R

def possible_zernike_polynomials():
    pass


def zernike_radial_singh(r, n):
    """Radial part of zernike polynomials. Modified Prata's algorithm."""
    R = np.ones((len(n), len(n), len(r))) * np.nan

    n_max = np.max(n)
    for n_i in range(n_max):
        R[n_i, n_i] = r**n_i

    for n_i in range(2, n_max + 2, 2):
        for m_i in range(0, n_max - n_i):
            K_1 = 2 * n_i / (m_i + n_i)
            K_2 = - (n_i - m_i) / (n_i + m_i)

            term_1 = r * K_1 * R[n_i + m_i - 1, abs(m_i - 1)]
            term_2 = K_2 * R[m_i + n_i - 2, m_i]

            R[n_i + m_i, m_i] = term_1 + term_2

    return R


def plot_zernike(r, l_modes, l, m):
    """Plot radial zernike with respect to radii."""
    algo = zernike_radial_singh(r, l_modes)[l, m]
    direct = zernike_direct_eval(r)[str(l) + str(m)]
    plt.plot(r, algo, label="computed")
    plt.plot(r, direct, label="analytic", linestyle="--")
    plt.title(f"{l=}, {m=}", fontsize=15)
    plt.ylabel(rf"$R_{m}^{l}$", fontsize=15)
    plt.xlabel("r", fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()


l_modes = np.arange(0, 9)
r = np.linspace(0, 1, 100)
l = 4
m = 0

algo = zernike_radial_singh(r, l_modes)[l, m]
direct = zernike_direct_eval(r)
plot_zernike(r, l_modes, l, m)

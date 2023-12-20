"""Classes and functions to be used for Zernike polynomial application."""

import functools
from abc import ABC, abstractmethod
from math import factorial

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
from jax.lax import fori_loop
from jax.scipy.special import gammaln


def custom_jvp(fun, *args, **kwargs):
    """Dummy function for custom_jvp without JAX."""
    fun.defjvp = lambda *args, **kwargs: None
    fun.defjvps = lambda *args, **kwargs: None
    return fun


class _Basis(ABC):
    """Basis is an abstract base class for spectral basis sets."""

    def __init__(self):
        self._enforce_symmetry()
        self._sort_modes()
        self._create_idx()
        # ensure things that should be ints are ints
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)
        self._NFP = int(self._NFP)
        self._modes = self._modes.astype(int)

    def _set_up(self):
        """Do things after loading or changing resolution."""
        self._enforce_symmetry()
        self._sort_modes()
        self._create_idx()
        # ensure things that should be ints are ints
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)
        self._NFP = int(self._NFP)
        self._modes = self._modes.astype(int)

    def _enforce_symmetry(self):
        """Enforce stellarator symmetry."""
        assert self.sym in [
            "sin",
            "sine",
            "cos",
            "cosine",
            "even",
            "cos(t)",
            False,
            None,
        ], f"Unknown symmetry type {self.sym}"
        if self.sym in ["cos", "cosine"]:  # cos(m*t-n*z) symmetry
            self._modes = self.modes[
                np.asarray(sign(self.modes[:, 1]) == sign(self.modes[:, 2]))
            ]
        elif self.sym in ["sin", "sine"]:  # sin(m*t-n*z) symmetry
            self._modes = self.modes[
                np.asarray(sign(self.modes[:, 1]) != sign(self.modes[:, 2]))
            ]
        elif self.sym == "even":  # even powers of rho
            self._modes = self.modes[np.asarray(self.modes[:, 0] % 2 == 0)]
        elif self.sym == "cos(t)":  # cos(m*t) terms only
            self._modes = self.modes[np.asarray(sign(self.modes[:, 1]) >= 0)]
        elif self.sym is None:
            self._sym = False

    def _sort_modes(self):
        """Sorts modes for use with FFT."""
        sort_idx = np.lexsort((self.modes[:, 1], self.modes[:, 0], self.modes[:, 2]))
        self._modes = self.modes[sort_idx]

    def _create_idx(self):
        """Create index for use with self.get_idx()."""
        self._idx = {}
        for idx, (L, M, N) in enumerate(self.modes):
            if L not in self._idx:
                self._idx[L] = {}
            if M not in self._idx[L]:
                self._idx[L][M] = {}
            self._idx[L][M][N] = idx

    def get_idx(self, L=0, M=0, N=0, error=True):
        """Get the index of the ``'modes'`` array corresponding to given mode numbers.

        Parameters
        ----------
        L : int
            Radial mode number.
        M : int
            Poloidal mode number.
        N : int
            Toroidal mode number.
        error : bool
            whether to raise exception if mode is not in basis, or return empty array

        Returns
        -------
        idx : ndarray of int
            Index of given mode numbers.

        """
        try:
            return self._idx[L][M][N]
        except KeyError as e:
            if error:
                raise ValueError(
                    "mode ({}, {}, {}) is not in basis {}".format(L, M, N, str(self))
                ) from e
            else:
                return np.array([]).astype(int)

    @abstractmethod
    def _get_modes(self):
        """ndarray: Mode numbers for the basis."""

    @abstractmethod
    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(3,)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of in, shape(num_modes,3), optional
            basis modes to evaluate (if None, full basis is used)
        unique : bool, optional
            whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """

    @abstractmethod
    def change_resolution(self):
        """Change resolution of the basis to the given resolutions."""

    @property
    def L(self):
        """int: Maximum radial resolution."""
        return self.__dict__.setdefault("_L", 0)

    @L.setter
    def L(self, L):
        assert int(L) == L, "Basis Resolution must be an integer!"
        self._L = int(L)

    @property
    def M(self):
        """int:  Maximum poloidal resolution."""
        return self.__dict__.setdefault("_M", 0)

    @M.setter
    def M(self, M):
        assert int(M) == M, "Basis Resolution must be an integer!"
        self._M = int(M)

    @property
    def N(self):
        """int: Maximum toroidal resolution."""
        return self.__dict__.setdefault("_N", 0)

    @N.setter
    def N(self, N):
        assert int(N) == N, "Basis Resolution must be an integer!"
        self._N = int(N)

    @property
    def NFP(self):
        """int: Number of field periods."""
        return self.__dict__.setdefault("_NFP", 1)

    @property
    def sym(self):
        """str: {``'cos'``, ``'sin'``, ``False``} Type of symmetry."""
        return self.__dict__.setdefault("_sym", False)

    @property
    def modes(self):
        """ndarray: Mode numbers [l,m,n]."""
        return self.__dict__.setdefault("_modes", np.array([]).reshape((0, 3)))

    @modes.setter
    def modes(self, modes):
        self._modes = modes

    @property
    def num_modes(self):
        """int: Total number of modes in the spectral basis."""
        return self.modes.shape[0]

    @property
    def spectral_indexing(self):
        """str: Type of indexing used for the spectral basis."""
        return self.__dict__.setdefault("_spectral_indexing", "linear")

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (L={}, M={}, N={}, NFP={}, sym={}, spectral_indexing={})".format(
                self.L, self.M, self.N, self.NFP, self.sym, self.spectral_indexing
            )
        )


class ZernikePolynomial(_Basis):
    """2D basis set for analytic functions in a unit disc.

    Parameters
    ----------
    L : int
        Maximum radial resolution. Use L=-1 for default based on M.
    M : int
        Maximum poloidal resolution.
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'ansi'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triangle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape.

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

    """

    def __init__(self, L, M, sym=False, spectral_indexing="ansi"):
        self.L = L
        self.M = M
        self.N = 0
        self._NFP = 1
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        self._modes = self._get_modes(
            L=self.L, M=self.M, spectral_indexing=self.spectral_indexing
        )

        super().__init__()

    def _get_modes(self, L=-1, M=0, spectral_indexing="ansi"):
        """Get mode numbers for Fourier-Zernike basis functions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        spectral_indexing : {``'ansi'``, ``'fringe'``}
            Indexing method, default value = ``'ansi'``

            For L=0, all methods are equivalent and give a "chevron" shaped
            basis (only the outer edge of the zernike pyramid of width M).
            For L>0, the indexing scheme defines order of the basis functions:

            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triangle shape. For L == M,
            the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
            to the bottom of the pyramid, increasing L while keeping M constant,
            giving a "house" shape.

            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape for L=2*M where
            the traditional fringe/U of Arizona indexing is recovered.
            For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

        """
        assert spectral_indexing in [
            "ansi",
            "fringe",
        ], "Unknown spectral_indexing: {}".format(spectral_indexing)
        default_L = {"ansi": M, "fringe": 2 * M}
        L = L if L >= 0 else default_L.get(spectral_indexing, M)
        self.L = L

        if spectral_indexing == "ansi":
            pol_posm = [
                [(m + d, m) for m in range(0, M + 1) if m + d < M + 1]
                for d in range(0, L + 1, 2)
            ]
            if L > M:
                pol_posm += [
                    (l, m)
                    for l in range(M + 1, L + 1)
                    for m in range(0, M + 1)
                    if (l - m) % 2 == 0
                ]

        elif spectral_indexing == "fringe":
            pol_posm = [
                [(m + d // 2, m - d // 2) for m in range(0, M + 1) if m - d // 2 >= 0]
                for d in range(0, L + 1, 2)
            ]
            if L > 2 * M:
                pol_posm += [
                    [(l - m, m) for m in range(0, M + 1)]
                    for l in range(2 * M, L + 1, 2)
                ]

        pol = [
            [(l, m), (l, -m)] if m != 0 else [(l, m)] for l, m in flatten_list(pol_posm)
        ]
        pol = np.array(flatten_list(pol))
        num_pol = len(pol)
        tor = np.zeros((num_pol, 1))

        return np.hstack([pol, tor])

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of int, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used).
        unique : bool, optional
            Whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff.

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            Basis functions evaluated at nodes.

        """
        if modes is None:
            modes = self.modes
        if derivatives[2] != 0:
            return jnp.zeros((nodes.shape[0], modes.shape[0]))
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        r, t, z = nodes.T
        l, m, n = modes.T
        lm = modes[:, :2]

        if unique:
            _, ridx, routidx = np.unique(
                r, return_index=True, return_inverse=True, axis=0
            )
            _, tidx, toutidx = np.unique(
                t, return_index=True, return_inverse=True, axis=0
            )
            _, lmidx, lmoutidx = np.unique(
                lm, return_index=True, return_inverse=True, axis=0
            )
            _, midx, moutidx = np.unique(
                m, return_index=True, return_inverse=True, axis=0
            )
            r = r[ridx]
            t = t[tidx]
            lm = lm[lmidx]
            m = m[midx]

        radial = zernike_radial(r[:, np.newaxis], lm[:, 0], lm[:, 1], dr=derivatives[0])
        poloidal = fourier(t[:, np.newaxis], m, 1, derivatives[1])

        if unique:
            radial = radial[routidx][:, lmoutidx]
            poloidal = poloidal[toutidx][:, moutidx]

        return radial * poloidal

    def change_resolution(self, L, M, sym=None):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        sym : bool
            Whether to enforce stellarator symmetry.

        Returns
        -------
        None

        """
        if L != self.L or M != self.M or sym != self.sym:
            self.L = L
            self.M = M
            self._sym = sym if sym is not None else self.sym
            self._modes = self._get_modes(
                self.L, self.M, spectral_indexing=self.spectral_indexing
            )
            self._set_up()


@functools.partial(jax.jit, static_argnums=3)
def zernike_radial(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Evaluates basis functions using JAX and a stable
    evaluation scheme based on jacobi polynomials and
    binomial coefficients. Generally faster for L>24
    and differentiable, but slower for low resolution.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    m = jnp.abs(m)
    alpha = m
    beta = 0
    n = (l - m) // 2
    s = (-1) ** n
    jacobi_arg = 1 - 2 * r**2
    if dr == 0:
        out = r**m * _jacobi(n, alpha, beta, jacobi_arg, 0)
    elif dr == 1:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        out = m * r ** jnp.maximum(m - 1, 0) * f - 4 * r ** (m + 1) * df
    elif dr == 2:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        out = (
            (m - 1) * m * r ** jnp.maximum(m - 2, 0) * f
            - 4 * (2 * m + 1) * r**m * df
            + 16 * r ** (m + 2) * d2f
        )
    elif dr == 3:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        d3f = _jacobi(n, alpha, beta, jacobi_arg, 3)
        out = (
            (m - 2) * (m - 1) * m * r ** jnp.maximum(m - 3, 0) * f
            - 12 * m**2 * r ** jnp.maximum(m - 1, 0) * df
            + 48 * (m + 1) * r ** (m + 1) * d2f
            - 64 * r ** (m + 3) * d3f
        )
    elif dr == 4:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        d3f = _jacobi(n, alpha, beta, jacobi_arg, 3)
        d4f = _jacobi(n, alpha, beta, jacobi_arg, 4)
        out = (
            (m - 3) * (m - 2) * (m - 1) * m * r ** jnp.maximum(m - 4, 0) * f
            - 8 * m * (2 * m**2 - 3 * m + 1) * r ** jnp.maximum(m - 2, 0) * df
            + 48 * (2 * m**2 + 2 * m + 1) * r**m * d2f
            - 128 * (2 * m + 3) * r ** (m + 2) * d3f
            + 256 * r ** (m + 4) * d4f
        )
    else:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )
    return s * jnp.where((l - m) % 2 == 0, out, 0)


@custom_jvp
@jax.jit
@jnp.vectorize
def _jacobi(n, alpha, beta, x, dx=0):
    """Jacobi polynomial evaluation.

    Implementation is only correct for non-negative integer coefficients,
    returns 0 otherwise.

    Parameters
    ----------
    n : int, array_like
        Degree of the polynomial.
    alpha : int, array_like
        Parameter
    beta : int, array_like
        Parameter
    x : float, array_like
        Points at which to evaluate the polynomial

    Returns
    -------
    P : ndarray
        Values of the Jacobi polynomial
    """
    # adapted from scipy:
    # https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/
    # scipy/special/orthogonal_eval.pxd#L144

    def _jacobi_body_fun(kk, d_p_a_b_x):
        d, p, alpha, beta, x = d_p_a_b_x
        k = kk + 1.0
        t = 2 * k + alpha + beta
        d = (
            (t * (t + 1) * (t + 2)) * (x - 1) * p + 2 * k * (k + beta) * (t + 2) * d
        ) / (2 * (k + alpha + 1) * (k + alpha + beta + 1) * t)
        p = d + p
        return (d, p, alpha, beta, x)

    n, alpha, beta, x = map(jnp.asarray, (n, alpha, beta, x))

    # coefficient for derivative
    c = (
        gammaln(alpha + beta + n + 1 + dx)
        - dx * jnp.log(2)
        - gammaln(alpha + beta + n + 1)
    )
    c = jnp.exp(c)
    # taking derivative is same as coeff*jacobi but for shifted n,a,b
    n -= dx
    alpha += dx
    beta += dx

    d = (alpha + beta + 2) * (x - 1) / (2 * (alpha + 1))
    p = d + 1
    d, p, alpha, beta, x = fori_loop(
        0, jnp.maximum(n - 1, 0).astype(int), _jacobi_body_fun, (d, p, alpha, beta, x)
    )
    out = _binom(n + alpha, n) * p
    # should be complex for n<0, but it gets replaced elsewhere so just return 0 here
    out = jnp.where(n < 0, 0, out)
    # other edge cases
    out = jnp.where(n == 0, 1.0, out)
    out = jnp.where(n == 1, 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (x - 1)), out)
    return c * out


def zernike_radial_optimized(x, l, m, beta=0):
    """Radial part of zernike polynomials.

    Evaluates basis functions using JAX and a stable
    evaluation scheme based on jacobi polynomials and
    binomial coefficients. Generally faster for L>24
    and differentiable, but slower for low resolution.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    m = np.abs(m)
    n = (l - m) // 2
    idx = np.lexsort((n, m))
    id0 = np.arange(0, len(l))
    id0 = id0[idx]

    l = l[idx]
    m = m[idx]
    n = n[idx]

    unique_values = np.unique(m)
    opt_param = []
    # For each unique value, find the maximum value in array2 where
    # this value occurs in array1
    for value in unique_values:
        indices = np.where(m == value)
        max_n = np.max(n[indices])
        opt_param.append(np.array([value, max_n]))
    opt_param = np.array(opt_param)
    m_opt = opt_param[:, 0]
    n_opt = opt_param[:, 1]

    # if we use JAX in future, this might becoma handy
    init = np.zeros((len(m), len(x)))

    # Broadcasting is used for element-wise operations
    result = zernike_radial_update(x, n_opt, m_opt, beta, init)
    result = np.where((l - m) % 2 == 0, result, 0)
    result = result[:, np.argsort(id0)]

    return result


def zernike_radial_update(x, n, alpha, beta, result):
    """Calculate the radial part of Zernike polynomial at points x.

    Parameters
    ----------
    x : array-like
        Points where the Jacobi polynomial is evaluated.
    n : array-like
        Degree of the Jacobi polynomial.
    alpha : array-like
        Alpha parameter of the Jacobi polynomial.
    beta : array-like
        Beta parameter of the Jacobi polynomial.

    Returns
    -------
    result : array
        Values of the Zernike polynomial at points x.

    """
    xj = 1 - 2 * x**2
    index = 0
    for i in range(len(alpha)):
        P_n1 = jacobi_poly_single(xj, 1, alpha[i], beta)
        P_n2 = jacobi_poly_single(xj, 0, alpha[i], beta)
        power = x ** alpha[i]
        result[index, :] = np.array((-1) ** 0 * power * P_n2)
        index += 1
        if n[i] >= 1:
            result[index, :] = np.array((-1) ** 1 * power * P_n1)
            index += 1
        if n[i] >= 2:
            for N in range(2, n[i] + 1):
                P_n = jacobi_poly_single(xj, N, alpha[i], beta, P_n1, P_n2)
                result[index, :] = np.array((-1) ** N * power * P_n)
                index += 1
                P_n2 = P_n1
                P_n1 = P_n
    return np.transpose(result)


def jacobi_poly_single(x, n, alpha, beta, P_n1=0, P_n2=0):
    """Evaluate Jacobi for single alpha and n pair."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2
    else:
        c = 2 * n + alpha + beta
        a1 = 2 * n * (c - n) * (c - 2)
        a2 = (c - 1) * (c * (c - 2) * x + (alpha - beta) * (alpha + beta))
        a3 = 2 * (n + alpha - 1) * (n + beta - 1) * c

        P_n = (a2 * P_n1 - a3 * P_n2) / a1

        return P_n
    

def zernike_radial_optimized_jit(x, l, m, beta=0):
    """Radial part of zernike polynomials.

    Evaluates basis functions using JAX and a stable
    evaluation scheme based on jacobi polynomials and
    binomial coefficients. Generally faster for L>24
    and differentiable, but slower for low resolution.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    m = np.abs(m)
    n = (l - m) // 2
    idx = np.lexsort((n, m))
    id0 = np.arange(0, len(l))
    id0 = id0[idx]

    l = l[idx]
    m = m[idx]
    n = n[idx]

    unique_values = np.unique(m)
    opt_param = []
    # For each unique value, find the maximum value in array2 where
    # this value occurs in array1
    for value in unique_values:
        indices = np.where(m == value)
        max_n = np.max(n[indices])
        opt_param.append(np.array([value, max_n]))
    opt_param = np.array(opt_param)
    m_opt = opt_param[:, 0]
    n_opt = opt_param[:, 1]

    # if we use JAX in future, this might becoma handy
    init = np.zeros((len(m), len(x)))

    # Broadcasting is used for element-wise operations
    result = zernike_radial_update_jit(x, n_opt, m_opt, beta, init)
    result = np.transpose(result)
    result = np.where((l - m) % 2 == 0, result, 0)
    result = result[:, np.argsort(id0)]

    return result


@jax.jit
def zernike_radial_update_jit(x, n, alpha, beta, result):
    """Calculate the radial part of Zernike polynomial at points x.

    Parameters
    ----------
    x : array-like
        Points where the Jacobi polynomial is evaluated.
    n : array-like
        Degree of the Jacobi polynomial.
    alpha : array-like
        Alpha parameter of the Jacobi polynomial.
    beta : array-like
        Beta parameter of the Jacobi polynomial.

    Returns
    -------
    result : array
        Values of the Zernike polynomial at points x.

    """

    def body(N, args):
        xj, alpha, beta, power, result, P_n1, P_n2, index = args
        P_n = jacobi_poly_single_jit(xj, N, alpha, beta, P_n1, P_n2)
        result = result.at[index, :].set(jnp.array((-1) ** N * power * P_n))
        index += 1
        P_n2 = P_n1
        P_n1 = P_n
        return (xj, alpha, beta, power, result, P_n1, P_n2, index)

    xj = 1 - 2 * x**2
    index = 0
    for i in range(alpha.size):
        m = alpha[i]
        P_n1 = jacobi_poly_single_jit(xj, 1, m, beta)
        P_n2 = jacobi_poly_single_jit(xj, 0, m, beta)
        power = x**m
        result = result.at[index, :].set(jnp.array((-1) ** 0 * power * P_n2))
        index += 1
        xj, m, beta, power, result, P_n1, P_n2, index = fori_loop(
            1,
            (n[i] + 1).astype(int),
            body,
            (xj, m, beta, power, result, P_n1, P_n2, index),
        )

    return result


def jacobi_poly_single_jit(x, n, alpha, beta, P_n1=0, P_n2=0):
    """Evaluate Jacobi for single alpha and n pair."""
    c = 2 * n + alpha + beta
    a1 = 2 * n * (c - n) * (c - 2)
    a2 = (c - 1) * (c * (c - 2) * x + (alpha - beta) * (alpha + beta))
    a3 = 2 * (n + alpha - 1) * (n + beta - 1) * c

    P_n = (a2 * P_n1 - a3 * P_n2) / a1
    P_n = jnp.where(n == 0, 1, P_n)
    P_n = jnp.where(n == 1, (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2, P_n)

    return P_n


def flatten_list(x, flatten_tuple=False):
    """Flatten a nested list.

    Parameters
    ----------
    x : list
        nested list of lists to flatten
    flatten_tuple : bool
        Whether to also flatten nested tuples.

    Returns
    -------
    x : list
        flattened input

    """
    types = (list,)
    if flatten_tuple:
        types += (tuple,)
    if isinstance(x, types):
        return [a for i in x for a in flatten_list(i, flatten_tuple)]
    else:
        return [x]


@jax.jit
def fourier(theta, m, NFP=1, dt=0):
    """Fourier series.

    Parameters
    ----------
    theta : ndarray, shape(N,)
        poloidal/toroidal coordinates to evaluate basis
    m : ndarray of int, shape(K,)
        poloidal/toroidal mode number(s)
    NFP : int
        number of field periods (Default = 1)
    dt : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    theta, m, NFP, dt = map(jnp.asarray, (theta, m, NFP, dt))
    m_pos = (m >= 0).astype(int)
    m_abs = jnp.abs(m) * NFP
    shift = m_pos * jnp.pi / 2 + dt * jnp.pi / 2
    return m_abs**dt * jnp.sin(m_abs * theta + shift)


@jax.jit
@jnp.vectorize
def _binom(n, k):
    """Binomial coefficient.

    Implementation is only correct for positive integer n,k and n>=k

    Parameters
    ----------
    n : int, array-like
        number of things to choose from
    k : int, array-like
        number of things chosen

    Returns
    -------
    val : int, float, array-like
        number of possible combinations
    """
    # adapted from scipy:
    # https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/
    # scipy/special/orthogonal_eval.pxd#L68

    n, k = map(jnp.asarray, (n, k))

    def _binom_body_fun(i, b_n):
        b, n = b_n
        num = n + 1 - i
        den = i
        return (b * num / den, n)

    kx = k.astype(int)
    b, n = fori_loop(1, 1 + kx, _binom_body_fun, (1.0, n))
    return b


def sign(x):
    """Sign function, but returns 1 for x==0.

    Parameters
    ----------
    x : array-like
        array of input values

    Returns
    -------
    y : array-like
        1 where x>=0, -1 where x<0

    """
    x = jnp.atleast_1d(x)
    y = jnp.where(x == 0, 1, jnp.sign(x))
    return y


def zernike_radial_coeffs(l, m, exact=True):
    """Polynomial coefficients for radial part of zernike basis.

    Parameters
    ----------
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    exact : bool
        whether to return exact coefficients with `object` dtype
        or return integer or floating point approximation

    Returns
    -------
    coeffs : ndarray
        Polynomial coefficients for Zernike polynomials, in descending powers of r.

    Notes
    -----
    Integer representation is exact up to l~54, so leaving `exact` arg as False
    can speed up evaluation with no loss in accuracy
    """
    l = np.atleast_1d(l).astype(int)
    m = np.atleast_1d(np.abs(m)).astype(int)
    lm = np.vstack([l, m]).T
    # for modest to large arrays, faster to find unique values and
    # only evaluate those
    lms, idx = np.unique(lm, return_inverse=True, axis=0)

    npoly = len(lms)
    lmax = np.max(lms[:, 0])
    coeffs = np.zeros((npoly, lmax + 1), dtype=object)
    lm_even = ((lms[:, 0] - lms[:, 1]) % 2 == 0)[:, np.newaxis]
    for ii in range(npoly):
        ll = lms[ii, 0]
        mm = lms[ii, 1]
        for s in range(mm, ll + 1, 2):
            coeffs[ii, s] = (
                (-1) ** ((ll - s) // 2)
                * factorial((ll + s) // 2)
                // (
                    factorial((ll - s) // 2)
                    * factorial((s + mm) // 2)
                    * factorial((s - mm) // 2)
                )
            )
    c = np.fliplr(np.where(lm_even, coeffs, 0))
    if not exact:
        try:
            c = c.astype(int)
        except OverflowError:
            c = c.astype(float)
    c = c[idx]
    return c


def zernike_radial_poly(r, l, m, dr=0, exact="auto"):
    """Radial part of zernike polynomials.

    Evaluates basis functions using numpy to
    exactly compute the polynomial coefficients
    and Horner's method for low resolution,
    or extended precision arithmetic for high resolution.
    Faster for low resolution, but not differentiable.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)
    exact : {"auto", True, False}
        Whether to use exact/extended precision arithmetic. Slower but more accurate.
        "auto" will use higher accuracy when needed.

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if exact == "auto":
        exact = np.max(l) > 54
    if exact:
        # this should give accuracy of ~1e-10 in the eval'd polynomials
        lmax = np.max(l)
        prec = int(0.4 * lmax + 8.4)
    else:
        prec = None
    coeffs = zernike_radial_coeffs(l, m, exact=exact)
    coeffs = polyder_vec(coeffs, dr, exact=exact)
    return polyval_vec(coeffs, r, prec=prec).T


def polyder_vec(p, m, exact=False):
    """Vectorized version of polyder.

    For differentiating multiple polynomials of the same degree

    Parameters
    ----------
    p : ndarray, shape(N,M)
        polynomial coefficients. Each row is 1 polynomial, in descending powers of x,
        each column is a power of x
    m : int >=0
        order of derivative
    exact : bool
        Whether to use exact integer arithmetic (not compatible with JAX, but may be
        needed for very high degree polynomials)

    Returns
    -------
    der : ndarray, shape(N,M)
        polynomial coefficients for derivative in descending order

    """
    if exact:
        return _polyder_exact(p, m)
    else:
        return _polyder_jax(p, m)


def _polyder_exact(p, m):
    factorial = np.math.factorial
    m = np.asarray(m, dtype=int)  # order of derivative
    p = np.atleast_2d(p)
    order = p.shape[1] - 1

    D = np.arange(order, -1, -1)
    num = np.array([factorial(i) for i in D], dtype=object)
    den = np.array([factorial(max(i - m, 0)) for i in D], dtype=object)
    D = (num // den).astype(p.dtype)

    p = np.roll(D * p, m, axis=1)
    idx = np.arange(p.shape[1])
    p = np.where(idx < m, 0, p)
    return p


@jax.jit
def _polyder_jax(p, m):
    p = jnp.atleast_2d(p)
    m = jnp.asarray(m).astype(int)
    order = p.shape[1] - 1
    D = jnp.arange(order, -1, -1)

    def body(i, Di):
        return Di * jnp.maximum(D - i, 1)

    D = fori_loop(0, m, body, jnp.ones_like(D))

    p = jnp.roll(D * p, m, axis=1)
    idx = jnp.arange(p.shape[1])
    p = jnp.where(idx < m, 0, p)

    return p


def polyval_vec(p, x, prec=None):
    """Evaluate a polynomial at specific values.

    Vectorized for evaluating multiple polynomials of the same degree.

    Parameters
    ----------
    p : ndarray, shape(N,M)
        Array of coefficient for N polynomials of order M.
        Each row is one polynomial, given in descending powers of x.
    x : ndarray, shape(K,)
        A number, or 1d array of numbers at
        which to evaluate p. If greater than 1d it is flattened.
    prec : int, optional
        precision to use, in number of decimal places. Default is
        double precision (~16 decimals) which should be enough for
        most cases with L <= 24

    Returns
    -------
    y : ndarray, shape(N,K)
        polynomials evaluated at x.
        Each row corresponds to a polynomial, each column to a value of x

    """
    if prec is not None and prec > 18:
        return _polyval_exact(p, x, prec)
    else:
        return _polyval_jax(p, x)


def _polyval_exact(p, x, prec):
    p = np.atleast_2d(p)
    x = np.atleast_1d(x).flatten()
    # TODO: possibly multithread this bit
    mpmath.mp.dps = prec
    y = np.array([np.asarray(mpmath.polyval(list(pi), x)) for pi in p])
    return y.astype(float)


@jax.jit
def _polyval_jax(p, x):
    p = jnp.atleast_2d(p)
    x = jnp.atleast_1d(x).flatten()
    npoly = p.shape[0]  # number of polynomials
    order = p.shape[1]  # order of polynomials
    nx = len(x)  # number of coordinates
    y = jnp.zeros((npoly, nx))

    def body(k, y):
        return y * x + jnp.atleast_2d(p[:, k]).T

    y = fori_loop(0, order, body, y)

    return y.astype(float)

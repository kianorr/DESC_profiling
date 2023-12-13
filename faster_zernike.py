"""zernike algorithms."""
from abc import ABC, abstractmethod

import numpy as np


class _PratasMethod(ABC):
    """
    Base class for using Prata's method following Singh's paper.

    Our `l` and `m` are equivalent to Singh's `p` and `q`, respectively.

    Attributes
    ----------
    R: `dict`
        Dictionary of numpy arrays for each derivative.
    """

    def __init__(self):
        """Init for base class."""
        # TODO: add l, m as attributes
        self.R = {}
        self.R[0] = None
        self.R[1] = None
        self.R[2] = None
        self.R[3] = None
        self.R[4] = None

    @abstractmethod
    def initial_condition(self, r, l_i):
        """Abstract method."""

    @abstractmethod
    def main_recurrence(self, K_1, K_2, r, l_i, m_i):
        """Abstract method."""

    @abstractmethod
    def zernike_radial_analytical(self, r):
        """Abstract method."""

    def calc_main_recurrence(self, K_1, K_2, r, l_i, m_i, dr=0):
        """Calculates the main recurrence relationship.

        This applies to each derivative, including the zeroth.

        Parameters
        ----------
        K_1: `float`
            A constant
        K_2: `float`
            A constant
        r: `np.ndarray`
            array of radii on the unit disc
        """
        # if else covering case where -1 becomes index
        term_1 = dr * K_1 * self.R[dr - 1][l_i + m_i - 1, abs(m_i - 1)] if dr > 0 else 0
        term_2 = K_1 * r * self.R[dr][l_i + m_i - 1, abs(m_i - 1)]
        term_3 = K_2 * self.R[dr][l_i + m_i - 2, m_i]
        self.R[dr][l_i + m_i, m_i] = term_1 + term_2 + term_3

    def zernike_radial_prata(self, r, n):
        """Radial part of zernike polynomials. Modified Prata's algorithm."""
        for dr in self.R.keys():
            self.R[dr] = np.ones((len(n), len(n), len(r))) * np.nan

        n_max = np.max(n)
        for l_i in range(n_max):
            self.initial_condition(r, l_i)

        for l_i in range(2, n_max + 2, 2):
            for m_i in range(0, n_max - l_i):
                K_1 = 2 * l_i / (m_i + l_i)
                K_2 = -(l_i - m_i) / (l_i + m_i)
                self.main_recurrence(K_1, K_2, r, l_i, m_i)


class ZerothDerivative(_PratasMethod):
    """Zernike radial."""

    def __init__(self):
        """Inherits base class."""
        # TODO: implement self._dr instead of inputting dr manually
        super().__init__()

    def initial_condition(self, r, l_i):
        """Calculates initial condition (l_i = m_i) for 0th derivative."""
        self.R[0][l_i, l_i] = r**l_i

    def main_recurrence(self, K_1, K_2, r, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().calc_main_recurrence(K_1, K_2, r, l_i, m_i, dr=0)

    def zernike_radial_analytical(self, r):
        """Calculate analytical expression for 0th derivative.

        From https://en.wikipedia.org/wiki/Zernike_polynomials#Radial_polynomials
        """
        R = {}
        R["42"] = 4 * r**4 - 3 * r**2
        R["31"] = 3 * r**3 - 2 * r
        R["20"] = 2 * r**2 - 1
        R["40"] = 6 * r**4 - 6 * r**2 + 1

        return R


class FirstDerivative(ZerothDerivative):
    """Zernike radial first derivative."""

    def __init__(self):
        """Inherits zeroth derivative."""
        super().__init__()

    def initial_condition(self, r, l_i):
        """Calculates initial condition (l_i = m_i) for 1st derivative."""
        super().initial_condition(r, l_i)
        self.R[1][l_i, l_i] = l_i * r ** (l_i - 1)

    def main_recurrence(self, K_1, K_2, r, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().main_recurrence(K_1, K_2, r, l_i, m_i)
        super().calc_main_recurrence(K_1, K_2, r, l_i, m_i, dr=1)

    def zernike_radial_analytical(self, r):
        """Calculate analytical expression for 3rd derivative."""
        R = {}
        R["42"] = 16 * r**3 - 6 * r
        R["31"] = 9 * r**2 - 2
        R["20"] = 4 * r
        R["40"] = 24 * r**3 - 12 * r

        return R


class SecondDerivative(FirstDerivative):
    """Zernike radial second derivative."""

    def __init__(self):
        """Inherits first derivative."""
        super().__init__()

    def initial_condition(self, r, l_i):
        """Calculates initial condition (l_i = m_i) for 2nd derivative."""
        super().initial_condition(r, l_i)
        self.R[2][l_i, l_i] = l_i * (l_i - 1) * r ** (l_i - 2)

    def main_recurrence(self, K_1, K_2, r, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().main_recurrence(K_1, K_2, r, l_i, m_i)
        super().calc_main_recurrence(K_1, K_2, r, l_i, m_i, dr=2)

    def zernike_radial_analytical(self, r):
        """Calculate analytical expression for 2nd derivative."""
        R = {}
        R["42"] = 48 * r**2 - 6
        R["31"] = 18 * r
        R["20"] = 4
        R["40"] = 48 * r**2 - 12

        return R


class ThirdDerivative(SecondDerivative):
    """Zernike radial third derivative."""

    def __init__(self):
        """Inherits second derivative."""
        super().__init__()

    def initial_condition(self, r, l_i):
        """Calculates initial condition (l_i = m_i) for 3rd derivative."""
        super().initial_condition(r, l_i)
        self.R[3][l_i, l_i] = l_i * (l_i - 1) * (l_i - 2) * r ** (l_i - 3)

    def main_recurrence(self, K_1, K_2, r, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().main_recurrence(K_1, K_2, r, l_i, m_i)
        super().calc_main_recurrence(K_1, K_2, r, l_i, m_i, dr=3)

    def zernike_radial_analytical(self, r):
        """Calculate analytical expression for 3rd derivative."""
        R = {}
        R["42"] = 96 * r
        R["31"] = 18
        R["20"] = 0
        R["40"] = 96 * r

        return R


class FourthDerivative(ThirdDerivative):
    """Zernike radial fourth derivative."""

    def __init__(self):
        """Inherits third derivative."""
        super().__init__()

    def initial_condition(self, r, l_i):
        """Calculates initial condition (l_i = m_i) for 3rd derivative."""
        super().initial_condition(r, l_i)
        self.R[4][l_i, l_i] = l_i * (l_i - 1) * (l_i - 2) * r ** (l_i - 3)

    def main_recurrence(self, K_1, K_2, r, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().main_recurrence(K_1, K_2, r, l_i, m_i)
        super().calc_main_recurrence(K_1, K_2, r, l_i, m_i, dr=4)

    def zernike_radial_analytical(self, r):
        """Calculate analytical expression for 4th derivative."""
        R = {}
        R["42"] = 96
        R["31"] = 0
        R["20"] = 0
        R["40"] = 96

        return R

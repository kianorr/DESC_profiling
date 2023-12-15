"""zernike algorithms."""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

IArray = NDArray[np.int64]
FArray = NDArray[np.float64]


class _PratasMethod(ABC):
    """
    Base class for using Prata's method following Singh's paper.

    Attributes
    ----------
    R: `dict`
        Dictionary of numpy arrays for each derivative.

    Notes
    -----
    Our `l` and `m` are equivalent to Singh's `p` and `q`, respectively.
    https://doi.org/10.1016/j.patcog.2010.02.005
    """

    def __init__(self, r: FArray, l: IArray):
        """Init for base class.

        Parameters
        ----------
        r: `np.ndarray`
            radii
        l: `np.ndarray`
            l modes (array of ints)
        """
        assert isinstance(r, np.ndarray), "r is an array."
        # TODO: assert that this is an array of ints
        assert isinstance(l, np.ndarray), "l is an array of ints."

        self._r = r
        self._l = l

        order = 4
        self.R = {
            dr: np.ones((len(l), len(l), len(r))) * np.nan for dr in range(order + 1)
        }

    @abstractmethod
    def initial_condition(self, l_i):
        """Abstract method."""

    @abstractmethod
    def main_recurrence(self, K_1, K_2, l_i, m_i):
        """Abstract method."""

    @abstractmethod
    def zernike_radial_analytical(self):
        """Abstract method."""

    def calc_main_recurrence(self, K_1, K_2, l_i, m_i, dr=0):
        """Calculates the main recurrence relationship.

        This applies to each derivative, including the zeroth.

        Parameters
        ----------
        K_1: `float`
            A constant
        K_2: `float`
            A constant
        l_i: `int`
            l mode
        m_i: `int`
            m mode
        dr: `int`
            derivative order
        """
        assert isinstance(dr, int), "derivative order is an integer."
        # if else covering case where -1 becomes index
        term_1 = dr * K_1 * self.R[dr - 1][l_i + m_i - 1, abs(m_i - 1)] if dr > 0 else 0
        term_2 = K_1 * self._r * self.R[dr][l_i + m_i - 1, abs(m_i - 1)]
        term_3 = K_2 * self.R[dr][l_i + m_i - 2, m_i]
        self.R[dr][l_i + m_i, m_i] = term_1 + term_2 + term_3

    def zernike_radial_prata(self):
        """Radial part of zernike polynomials. Modified Prata's algorithm."""
        n_max = np.max(self._l)
        for l_i in range(n_max):
            self.initial_condition(l_i)

        for l_i in range(2, n_max + 2, 2):
            for m_i in range(0, n_max - l_i):
                K_1 = 2 * l_i / (m_i + l_i)
                K_2 = -(l_i - m_i) / (l_i + m_i)
                self.main_recurrence(K_1, K_2, l_i, m_i)


class ZerothDerivative(_PratasMethod):
    """Zernike radial."""

    def __init__(self, r, l):
        """Inherits base class."""
        # TODO: implement self._dr instead of inputting dr manually
        super().__init__(r, l)
        self.__dr = 0

    def initial_condition(self, l_i):
        """Calculates initial condition (l_i = m_i) for 0th derivative."""
        self.R[self.__dr][l_i, l_i] = self._r**l_i

    def main_recurrence(self, K_1, K_2, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().calc_main_recurrence(K_1, K_2, l_i, m_i, dr=self.__dr)

    def get_zeroth_derivative(self):
        """Getter."""
        return self.R[self.__dr]

    def zernike_radial_analytical(self):
        """Calculate analytical expression for 0th derivative.

        From https://en.wikipedia.org/wiki/Zernike_polynomials#Radial_polynomials
        """
        R = {}
        R["42"] = 4 * self._r**4 - 3 * self._r**2
        R["31"] = 3 * self._r**3 - 2 * self._r
        R["20"] = 2 * self._r**2 - 1
        R["40"] = 6 * self._r**4 - 6 * self._r**2 + 1

        return R


class FirstDerivative(ZerothDerivative):
    """Zernike radial first derivative."""

    def __init__(self, r, l):
        """Inherits zeroth derivative."""
        super().__init__(r, l)
        self.__dr = 1

    def initial_condition(self, l_i):
        """Calculates initial condition (l_i = m_i) for 1st derivative."""
        super().initial_condition(l_i)
        self.R[self.__dr][l_i, l_i] = l_i * self._r ** (l_i - 1)

    def main_recurrence(self, K_1, K_2, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().main_recurrence(K_1, K_2, l_i, m_i)
        super().calc_main_recurrence(K_1, K_2, l_i, m_i, dr=1)

    def get_first_derivative(self):
        """Getter."""
        return self.R[self.__dr]

    def zernike_radial_analytical(self):
        """Calculate analytical expression for 3rd derivative."""
        R = {}
        R["42"] = 16 * self._r**3 - 6 * self._r
        R["31"] = 9 * self._r**2 - 2
        R["20"] = 4 * self._r
        R["40"] = 24 * self._r**3 - 12 * self._r

        return R


class SecondDerivative(FirstDerivative):
    """Zernike radial second derivative."""

    def __init__(self, r, l):
        """Inherits first derivative."""
        super().__init__(r, l)
        self.__dr = 2

    def initial_condition(self, l_i):
        """Calculates initial condition (l_i = m_i) for 2nd derivative."""
        super().initial_condition(l_i)
        self.R[self.__dr][l_i, l_i] = l_i * (l_i - 1) * self._r ** (l_i - 2)

    def main_recurrence(self, K_1, K_2, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().main_recurrence(K_1, K_2, l_i, m_i)
        super().calc_main_recurrence(K_1, K_2, l_i, m_i, dr=2)

    def get_second_derivative(self):
        """Getter."""
        return self.R[self.__dr]

    def zernike_radial_analytical(self):
        """Calculate analytical expression for 2nd derivative."""
        R = {}
        R["42"] = 48 * self._r**2 - 6
        R["31"] = 18 * self._r
        R["20"] = 4
        R["40"] = 48 * self._r**2 - 12

        return R


class ThirdDerivative(SecondDerivative):
    """Zernike radial third derivative."""

    def __init__(self, r, l):
        """Inherits second derivative."""
        super().__init__(r, l)
        self.__dr = 3

    def initial_condition(self, l_i):
        """Calculates initial condition (l_i = m_i) for 3rd derivative."""
        super().initial_condition(l_i)
        self.R[self.__dr][l_i, l_i] = l_i * (l_i - 1) * (l_i - 2) * self._r ** (l_i - 3)

    def main_recurrence(self, K_1, K_2, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().main_recurrence(K_1, K_2, l_i, m_i)
        super().calc_main_recurrence(K_1, K_2, l_i, m_i, dr=self.__dr)

    def get_third_derivative(self):
        """Getter."""
        return self.R[self.__dr]

    def zernike_radial_analytical(self):
        """Calculate analytical expression for 3rd derivative."""
        R = {}
        R["42"] = 96 * self._r
        R["31"] = 18
        R["20"] = 0
        R["40"] = 96 * self._r

        return R


class FourthDerivative(ThirdDerivative):
    """Zernike radial fourth derivative."""

    def __init__(self, r, l):
        """Inherits third derivative."""
        super().__init__(r, l)
        self.__dr = 4

    def initial_condition(self, l_i):
        """Calculates initial condition (l_i = m_i) for 3rd derivative."""
        super().initial_condition(l_i)
        self.R[self.__dr][l_i, l_i] = l_i * (l_i - 1) * (l_i - 2) * self._r ** (l_i - 3)

    def main_recurrence(self, K_1, K_2, l_i, m_i):
        """Calculates main recurring with specification of derivative."""
        super().main_recurrence(K_1, K_2, l_i, m_i)
        super().calc_main_recurrence(K_1, K_2, l_i, m_i, dr=self.__dr)

    def get_fourth_derivative(self):
        """Getter."""
        return self.R[self.__dr]

    def zernike_radial_analytical(self):
        """Calculate analytical expression for 4th derivative."""
        R = {}
        R["42"] = 96
        R["31"] = 0
        R["20"] = 0
        R["40"] = 96

        return R

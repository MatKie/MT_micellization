import numpy as np


class BaseMicelle(object):
    """
    Base class for all shapes of micelles
    """

    def __init__(self, g, T, nt):
        self._g = g
        self._T = T
        self._nt = nt

    @property
    def volume(self):
        v_ch3 = 54.6 + 0.124 * (self.T - 298)
        v_ch2 = 26.9 + 0.0146 * (self.T - 298)

        _volume = v_ch3 + (self.nt - 1) * v_ch2
        return _volume

    @property
    def length(self):
        return 1.5 + 1.265 * self.nt

    @property
    def g(self):
        return self._g

    @property
    def T(self):
        return self._T

    @property
    def nt(self):
        return self._nt

    @g.setter
    def g(self, new_g):
        if new_g > 0:
            self._g = new_g
        else:
            raise ValueError("Aggregation size must be greater than zero.")

    @T.setter
    def T(self, new_T):
        if new_T > 0:
            self._T = new_T
        else:
            raise ValueError("Temperature needs to be positive.")

    @nt.setter
    def nt(self, new_nt):
        if new_nt > 0:
            self._nt = new_nt
        else:
            raise ValueError("Taillenth needs to be greater than zero.")


class SphericalMicelle(BaseMicelle):
    """
    Class for the calculation of chemical potential incentive of 
    spherical micelles. 
    """

    @property
    def radius(self):
        return np.cbrt(3 * self.g * self.volume / (4.0 * np.pi))

    @property
    def area_per_surfactant(self):
        return 4.0 * np.pi * self.radius * self.radius / self.g

    @property
    def area(self):
        return self.area_per_surfactant * self.g


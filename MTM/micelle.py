import numpy as np


class BaseMicelle(object):
    """
    Base class for all shapes of micelles
    """

    def __init__(self, g, T, nt):
        self.g = g
        self.T = T
        self.nt = nt
        self.transfer_free_energy = None

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
            raise ValueError("Tail length needs to be greater than zero.")

    def get_transfer_free_energy(self, method="empirical"):
        methods = {"empirical": self._transfer_empirical, "SAFT": None}
        self.transfer_free_energy = methods.get(method, self._transfer_empirical())()
        return self.transfer_free_energy

    def _transfer_empirical(self):
        transfer_ch3 = (
            3.38 * np.log(self.T) + 4064.0 / self.T + 0.02595 * self.T - 44.13
        )

        transfer_ch2 = 5.85 * np.log(self.T) + 896.0 / self.T - 0.0056 * self.T - 36.15
        return (self.nt - 1.0) * transfer_ch2 + transfer_ch3


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


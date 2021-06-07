import numpy as np
from abc import ABC, abstractmethod


class BaseMicelle(object):
    """
    Base class for all shapes of micelles
    """

    def __init__(
        self, g, T, nt,
    ):
        self.g = g
        self.T = T
        self.nt = nt
        self.transfer_free_energy = None
        self.interface_free_energy = None

    @property
    @abstractmethod
    def area_per_surfactant(self):
        """
        Geometrical surface area of respective micelle shape over
        the number of surfactants in the shape.
        """

    @property
    def segment_length(self):
        return 0.46  # nm

    @property
    def shielded_surface_area_per_surfactant(self):

        return self.segment_length * self.segment_length

    @property
    def volume(self):
        v_ch3 = 54.6 + 0.124 * (self.T - 298)
        v_ch2 = 26.9 + 0.0146 * (self.T - 298)

        _volume = v_ch3 + ((self.nt - 1) * v_ch2)
        _volume /= 1000.0  # Angstrom^3 -> nm^3
        return _volume

    @property
    def length(self):
        # in nm
        return (1.5 + 1.265 * self.nt) / 10.0

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, new_g):
        if new_g > 0:
            self._g = new_g
        else:
            raise ValueError("Aggregation size must be greater than zero.")

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, new_T):
        if new_T > 0:
            self._T = new_T
        else:
            raise ValueError("Temperature needs to be positive.")

    @property
    def nt(self):
        return self._nt

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

    def get_interface_free_energy(self, method="empirical", curvature="flat"):
        methods = {"flat": self._interface_flat, "curved": None}
        self.interface_free_energy = methods.get(curvature, self._interface_flat)(
            method=method
        )

        return self.interface_free_energy

    def _interface_flat(self, method="empirical"):
        methods = {"empirical": self._sigma_agg, "SAFT": None}
        sigma_agg = methods.get(method, self._sigma_agg)()

        return sigma_agg * (
            self.area_per_surfactant - self.shielded_surface_area_per_surfactant
        )

    def _sigma_agg(self):
        M = (self.nt - 1) * 14.0266 + 15.035
        sigma_o = 35.0 - (325.0 / np.cbrt(M * M)) - (0.098 * (self.T - 298.0))
        sigma_w = 72.0 - (0.16 * (self.T - 298.0))

        sigma_agg = sigma_o + sigma_w - (1.1 * np.sqrt(sigma_o * sigma_w))
        # mN/m (or mJ/m^2) to J/nm^2 and w. kbT to 1/nm^2 / kbT
        sigma_agg /= 1.38064852 * 0.01 * self.T
        return sigma_agg


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


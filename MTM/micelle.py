import numpy as np
from abc import ABC, abstractmethod
import warnings

class BaseMicelle(object):
    """
    This class serves as base to calculate the free energy difference
    between a surfactant in a micelle of certain aggregation size and
    the surfactant dispersed in an aqueous solution. 
    See:
    S.Enders and D.Haentzschel: Fluid Phase Equilibria 153 1998 1–21 Z.
                               Thermodynamics
    """

    def __init__(
        self, surfactants_number, temperature, tail_carbons, 
        headgroup_area=0.49
    ):
        """
        Parameters
        ----------
        surfactants_number : float
            Number of surfactants per micelle. Only integer values 
            are sensible, but float for sake of optimisation.
        temperature : float
            system temperature.
        tail_carbons : float
            Number of carbons in the surfactant tail. Only integer 
            values are sensible, but float for sake of optimisation.
        headgroup_area : float
            Headgroup cross sectional area in nm^2. 
            Optional, default 0.49.
        Attributes:
        -----------

        """
        self.surfactants_number = surfactants_number
        self.temperature = temperature
        self.tail_carbons = tail_carbons
        self.headgroup_area = headgroup_area
        self.transfer_free_energy = None
        self.interface_free_energy = None
        # Exponent is 10^-23, account for it in conversions!
        self.boltzman = 1.38064852

    @property
    @abstractmethod
    def area_per_surfactant(self):
        """
        Geometrical surface area of respective micelle shape over
        the number of surfactants in the shape.
        """

    @property
    def segment_length(self):
        """
        Segment length as used in Nagarajan's approach on deformation
        free energy. In nm.
        """
        return 0.46  # nm

    @property
    def shielded_surface_area_per_surfactant(self):
        """
        Square of the segment length. In nm^2.
        """
        return self.segment_length * self.segment_length

    @property
    def volume(self):
        """
        Volume of a carbon tail depending on the # of carbbson. In nm^3.

        V = V_ch3 + (tail_carbons - 1) * V_ch2
        """
        v_ch3 = 54.6 + 0.124 * (self.temperature - 298)
        v_ch2 = 26.9 + 0.0146 * (self.temperature - 298)

        _volume = v_ch3 + ((self.tail_carbons - 1) * v_ch2)
        _volume /= 1000.0  # Angstrom^3 -> nm^3
        return _volume

    @property
    def length(self):
        """
        Extended length of the surfactant tail dep. on # Carbons. In nm.
        """
        return (1.5 + 1.265 * self.tail_carbons) / 10.0

    @property
    def surfactants_number(self):
        """
        Number of surfactants in the micelle.
        """
        return self._g

    @surfactants_number.setter
    def surfactants_number(self, new_g):
        if new_g > 0:
            self._g = new_g
        else:
            raise ValueError("Aggregation size must be greater than zero.")

    @property
    def temperature(self):
        """
        Temperature of the system.
        """
        return self._T

    @temperature.setter
    def temperature(self, new_T):
        if new_T > 0:
            self._T = new_T
        else:
            raise ValueError("Temperature needs to be positive.")

    @property
    def tail_carbons(self):
        """
        Number of carbons in the surfactant tail.
        """
        return self._nt

    @tail_carbons.setter
    def tail_carbons(self, new_nt):
        if new_nt > 0:
            self._nt = new_nt
        else:
            raise ValueError("Tail length needs to be greater than zero.")

    @property
    def headgroup_area(self):
        '''
        Headgroup area for steric free energy in nm^2.
        '''
        return self._ap 

    @headgroup_area.setter
    def headgroup_area(self, new_ap):
        if new_ap > 1:
            warnings.warn(UserWarning('Headgroup area seems unusually large \
                -- needs to be in nm^2.'))
        if new_ap > 0:
            self._ap = new_ap
        else:
            raise ValueError('Headgroup area needs to be greater than zero.')

    def get_transfer_free_energy(self, method="empirical"):
        """
        High level function to get transfer free energy. 
        Dependent only on number of carbons in tail.

        Parameters
        ----------
        method : str, optional
            whether to use empirical correlation or a SAFT approach, by default "empirical".

        Returns
        -------
        float
            Transfer free energy in units of k_b*T.
        """
        methods = {"empirical": self._transfer_empirical}
        self.transfer_free_energy = methods.get(method)
        if self.transfer_free_energy is None:
            self._raise_not_implemented(methods)

        return self.transfer_free_energy()

    def _transfer_empirical(self):
        """
        Correlation taken from Enders and Haentzschel 1998.
        """
        transfer_ch3 = (
            3.38 * np.log(self.temperature)
            + 4064.0 / self.temperature
            - 44.13
            + 0.02595 * self.temperature
        )

        transfer_ch2 = (
            5.85 * np.log(self.temperature)
            + 896.0 / self.temperature
            - 36.15
            - 0.0056 * self.temperature
        )
        return (self.tail_carbons - 1.0) * transfer_ch2 + transfer_ch3

    def get_interface_free_energy(self, method="empirical", curvature="flat"):
        """
        High level function to get interfacial free energy.

        Parameters
        ----------
        method : str, optional
            whether to use empirical correlation or a SAFT approach, by default "empirical"
        curvature : str, optional
            Whether to use a curvature correction or not ('flat'), by default "flat"

        Returns
        -------
        float
            Interface free energy in units of k_b*T.
        """
        methods = {"flat": self._interface_flat}
        self.interface_free_energy = methods.get(curvature)
        if self.interface_free_energy is None:
            self._raise_not_implemented(methods)

        return self.interface_free_energy(method=method)

    def _interface_flat(self, method="empirical"):
        """
        Low level function to calculate interfacial energy of flat
        interface:

         mu/kT = sig_agg/kT * (a - a_0)

        where a_0 is the 'shielded' area per headgroup and a the total
        area per surfactant.

        Parameters
        ----------
        method : str, optional
            whether to use sig_agg from empirical correlation or other means, by default "empirical"

        Returns
        -------
        float
            Interface free energy in units of k_b*T.
        """
        methods = {"empirical": self._sigma_agg}
        sigma_agg = methods.get(method)
        if sigma_agg is None:
            self._raise_not_implemented(methods)

        return sigma_agg() * (
            self.area_per_surfactant - self.shielded_surface_area_per_surfactant
        )

    def _sigma_agg(self):
        """
        Correlation for interfacial tension oil-water from Enders and Haentzschel 1998
        """
        M = (self.tail_carbons - 1) * 14.0266 + 15.035
        sigma_o = 35.0 - (325.0 / np.cbrt(M * M)) - (0.098 * (self.temperature - 298.0))
        sigma_w = 72.0 - (0.16 * (self.temperature - 298.0))

        sigma_agg = sigma_o + sigma_w - (1.1 * np.sqrt(sigma_o * sigma_w))
        # mN/m (or mJ/m^2) to J/nm^2 and w. kbT to 1/nm^2 / kbT
        sigma_agg /= self.boltzman * 0.01 * self.temperature
        return sigma_agg

    def get_steric_free_energy(self, method='VdW'):
        methods = {"VdW": self._steric_vdw}
        _free_energy_method = methods.get(method)
        if _free_energy_method is None:
            self._raise_not_implemented(methods)

        self.steric_free_energy = _free_energy_method()
        return self.steric_free_energy

    @abstractmethod
    def _steric_vdw(self):
        '''
        Steric free energy from VdW approach.
        Returns a float, free energy in kT. 
        '''
    
    def _raise_not_implemented(self, methods):
        error_string = "Only these methods are implemented: {:s}".format(
            *methods.keys()
        )
        raise NotImplementedError(error_string)


class SphericalMicelle(BaseMicelle):
    """
    Class for the calculation of chemical potential incentive of 
    spherical micelles. 
    """

    @property
    def radius(self):
        """
        Radius of a sphere as given in Enders and Haentzschel 1998.
        """
        return np.cbrt(3 * self.surfactants_number * self.volume / (4.0 * np.pi))

    @property
    def area_per_surfactant(self):
        """
        Area per surfactant in a spherical micelle of given aggregation
        number and # of carbons in the tail and temperature.
        """
        radius = self.radius
        return 4.0 * np.pi * radius * radius / self.surfactants_number

    @property
    def area(self):
        """
        Area of a spherical micelle of given aggregation
        number and # of carbons in the tail and temperature.
        """
        return self.area_per_surfactant * self.surfactants_number

    def _steric_vdw(self):
        _headgroup_area = self.headgroup_area
        _area_per_surfactant = self.area_per_surfactant
        if _headgroup_area >= _area_per_surfactant:
            raise ValueError('headgroup area larger than area \
                per surfactant.')
        else:
            return -np.log(1 - (_headgroup_area / _area_per_surfactant))
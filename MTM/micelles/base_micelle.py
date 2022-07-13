import numpy as np
from abc import ABC, abstractmethod
import warnings
from scipy.optimize import minimize, LinearConstraint
from ._transfer_saft import TransferSaft
from ._transfer_saft_gamma import TransferSaftGamma
from ._sigma_sgt import SigmaSGT


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
        self,
        surfactants_number,
        temperature,
        tail_carbons,
        headgroup_area=0.49,
        **kwargs
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
        self.deformation_free_energy = None
        self.steric_free_energy = None
        self.delta_chempot = None
        # Exponent is 10^-23, account for it in conversions!
        self.boltzman = 1.38064852
        self._transfer_saft = TransferSaft()
        self._transfer_saft_gamma = TransferSaftGamma(self.tail_carbons)
        self._sigma_sgt = SigmaSGT(self.tail_carbons)

    def get_delta_chempot(
        self,
        transfer_method="empirical",
        interface_method="empirical",
        curvature="flat",
        deformation_method="nagarajan",
        steric_method="VdW",
    ):
        """
        Call all contributions to chemical potential incentive
        towards micellisation and sum up.
        Contributions are: transfer, interface, deformation and
                           steric.

        Parameters
        ----------
        transfer_method: str, optional
            Which method used to calculate transfer free energy. By
            defualt 'empirical'.
        interface_method: str, optional
            Which method used to calculate contribution from interface
            formation. By default 'empirical'.
        curvature: str, optional
            Whether or not to apply a curvature correction to the
            interface contribution. By default 'flat'.
        deformation_method: str, optional
            Which method to use for deformation free energy. By default
            'nagarajan'.
        steric_method: str, optional
            Which method to use for steric interactions of headgroups.
            By default 'VdW'.

        Returns
        -------
        float
            chemical potential difference in k_b * T.
        """
        self.transfer_free_energy = self.get_transfer_free_energy(
            method=transfer_method
        )
        self.interface_free_energy = self.get_interface_free_energy(
            method=interface_method, curvature=curvature
        )
        self.deformation_free_energy = self.get_deformation_free_energy(
            method=deformation_method
        )
        self.steric_free_energy = self.get_steric_free_energy(method=steric_method)

        self.delta_chempot = (
            self.transfer_free_energy
            + self.interface_free_energy
            + self.deformation_free_energy
            + self.steric_free_energy
        )

        return self.delta_chempot

    @property
    @abstractmethod
    def area_per_surfactant(self):
        """
        Geometrical surface area of respective micelle shape over
        the number of surfactants in the shape.
        """

    @property
    def geometry_check(self):
        return True

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
        """
        Headgroup area for steric free energy in nm^2.
        """
        return self._ap

    @headgroup_area.setter
    def headgroup_area(self, new_ap):
        if new_ap > 1:
            warnings.warn(
                UserWarning(
                    "Headgroup area seems unusually large \
                -- needs to be in nm^2."
                )
            )
        if new_ap > 0:
            self._ap = new_ap
        else:
            raise ValueError("Headgroup area needs to be greater than zero.")

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
        methods = {
            "empirical": self._transfer_empirical,
            "assoc_saft": self._transfer_assoc_saftvrmie,
            "empirical_saft": self._empirical_saft_transfer,
            "assoc_saft_gamma": self._transfer_assoc_saftgammamie,
            "assoc_saft_gamma_hydrated": self._transfer_assoc_saftgammamie_hydrated,
            "assoc_saft_gamma_alkyl": self._transfer_assoc_saftgammamie_alkyl,
        }
        _free_energy_method = methods.get(method)
        if _free_energy_method is None:
            self._raise_not_implemented(methods)

        self.transfer_free_energy = _free_energy_method()
        return self.transfer_free_energy

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

    def _transfer_assoc_saftvrmie(self):
        """
        Correlation based on SAFT VR Mie model with associating water
        model
        """
        self._transfer_saft.temperature = self.temperature
        nc = self.tail_carbons
        mu_ch2 = self._transfer_saft.mu_ch2
        mu_alkanes = self._transfer_saft.mu_alkanes
        if nc < 5:
            raise ValueError("Probably not sensible to have alkane tails that small?")
        elif nc > 4 and nc < 12:
            mu_alkane = mu_alkanes.get(nc)
        elif nc > 11:
            mu_alkane = mu_alkanes.get(11) + (nc - 11) * mu_ch2

        mu = 0.5 * mu_alkane + ((nc - 2.0) / 2.0 * mu_ch2)

        return mu

    def _empirical_saft_transfer(self):
        """
        Use \Delta\mu_CH3 + (C_n - 1) \Delta\mu_CH2 = \Delta\mu_g
        """
        self._transfer_saft.temperature = self.temperature
        nc = self.tail_carbons
        mu_ch2 = self._transfer_saft.mu_ch2
        mu_alkanes = self._transfer_saft.mu_alkanes
        if nc < 5:
            raise ValueError("Probably not sensible to have alkane tails that small?")
        elif nc > 4 and nc < 12:
            mu_alkane = mu_alkanes.get(nc)
        elif nc > 11:
            mu_alkane = mu_alkanes.get(11) + (nc - 11) * mu_ch2

        mu_ch3 = (mu_alkanes.get(nc) - (nc - 2) * mu_ch2) / 2
        mu = mu_ch3 + (nc - 1) * mu_ch2

        return mu

    def _transfer_assoc_saftgammamie(self):
        """
        Correlation based on Saft Gamma Mie model with associating water model
        but working just like the saft-vr mie based approach
        """
        # temperature setter calls the calculation routines
        self._transfer_saft_gamma.alkyl_ends = False
        self._transfer_saft_gamma.hydrate_c1_carbon = False
        self._transfer_saft_gamma.temperature = self.temperature
        mu_ch2 = self._transfer_saft_gamma.mu_ch2
        mu_alkane = self._transfer_saft_gamma.mu_alkane

        mu = 0.5 * mu_alkane + ((self.tail_carbons - 2.0) / 2.0 * mu_ch2)
        return mu

    def _transfer_assoc_saftgammamie_hydrated(self):
        """
        Correlation based on Saft Gamma Mie model with associating water model.
        Calculates the transfer contribution of a (m-1) alkyl tail.
        If we got a surfactant with a nonyl tail, this will calculate
        for an octyl tail (1 CH3 + 7 CH2).
        """
        # temperature setter calls the calculation routines
        self._transfer_saft_gamma.alkyl_ends = True
        self._transfer_saft_gamma.hydrate_c1_carbon = True
        self._transfer_saft_gamma.temperature = self.temperature
        mu_alkane = self._transfer_saft_gamma.mu_alkane

        return mu_alkane

    def _transfer_assoc_saftgammamie_alkyl(self):
        """
        Correlation based on Saft Gamma Mie model with associating water model.
        Calculates the transfer contribution of a (m) alkyl tail.
        If we got a surfactant with a nonyl tail, this will calculate
        for an nonyl tail (1 CH3 + 8 CH2).
        """
        # temperature setter calls the calculation routines
        self._transfer_saft_gamma.alkyl_ends = True
        self._transfer_saft_gamma.hydrate_c1_carbon = False
        self._transfer_saft_gamma.temperature = self.temperature
        mu_alkane = self._transfer_saft_gamma.mu_alkane

        return mu_alkane

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
        _free_energy_method = methods.get(curvature)
        if _free_energy_method is None:
            self._raise_not_implemented(methods)

        self.interface_free_energy = _free_energy_method(method=method)
        return self.interface_free_energy

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
        methods = {"empirical": self._sigma_agg, "sgt": self._sigma_sgt}
        sigma_agg = methods.get(method)
        if sigma_agg is None:
            self._raise_not_implemented(methods)

        return sigma_agg() * (
            self.area_per_surfactant - self.shielded_surface_area_per_surfactant
        )

    def _sigma_sgt(self):
        """
        Interfacial tension from SGT with SAFT-VR Mie
        """
        self._sigma_sgt.temperature = self.temperature
        sigma_sgt = self._sigma_sgt.get_ift()

        return self._sigma_agg()

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

    def get_steric_free_energy(self, method="VdW"):
        """
        High level steric free energy getter.

        Parameters
        ----------
        method: str, optional
            Which method to use, by default van der Waals.
        Returns
        -------
        float
            Steric free energy in k_b * T.
        """
        methods = {"VdW": self._steric_vdw}
        _free_energy_method = methods.get(method)
        if _free_energy_method is None:
            self._raise_not_implemented(methods)

        self.steric_free_energy = _free_energy_method()
        return self.steric_free_energy

    @abstractmethod
    def _steric_vdw(self):
        """
        Steric free energy from VdW approach.
        Returns a float, free energy in kT.
        """

    def get_deformation_free_energy(self, method="nagarajan"):
        """
        High level deformation free energy getter.

        Parameters
        ----------
        method: str, optional
            which method to use in calculation, by default nagarajan.

        Returns
        -------
        float
            Free energy in k_b * T.
        """
        methods = {"nagarajan": self._deformation_nagarajan}
        _free_energy_method = methods.get(method)
        if _free_energy_method is None:
            self._raise_not_implemented(methods)

        self.deformation_free_energy = _free_energy_method()
        return self.deformation_free_energy

    @abstractmethod
    def _deformation_nagarajan(self):
        """
        Method to calculate deformation free energy following
        Nagarajan. Dependent on shape of micelle.
        """

    def _raise_not_implemented(self, methods):
        error_string = "Only these methods are implemented: {:s}".format(
            ", ".join(methods.keys())
        )
        raise NotImplementedError(error_string)

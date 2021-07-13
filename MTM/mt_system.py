from .micelles.spherical_micelle import SphericalMicelle
from .micelles.rodlike_micelle import RodlikeMicelle
from .micelles.globular_micelle import GlobularMicelle
from .micelles.bilayer_vesicle import BilayerVesicle
import numpy as np
from scipy.optimize import newton, root_scalar
from functools import reduce


class MTSystem(object):
    """
    High level class to calculate the aggregate distribution
    of a water - surfactant mixture (single surfactant).
    Following Enders and Haentzschel, 1998
    """

    types = {
        "spheres": SphericalMicelle,
        "globular": GlobularMicelle,
        "rodlike": RodlikeMicelle,
        "vesicles": BilayerVesicle,
    }

    def __init__(
        self,
        T=298.15,
        m=8,
        surfactant_concentration=None,
        spheres=True,
        globular=False,
        rodlike=True,
        vesicles=True,
    ):
        self.free_energy_minimas = None
        self.free_energy_types = None
        self.monomer_concentration = None
        self.wanted_keys = None  # set by next function
        self.micelles = self.get_micelles(
            T, m, spheres=spheres, globular=globular, rodlike=rodlike, vesicles=vesicles
        )
        self.surfactant_concentration = surfactant_concentration
        self.T = T
        self.m = m

    def get_free_energy_minimas(self, T=None, m=None, **kwargs):
        """
        Calculate free energy of various micelle shapes over all
        aggregatin numbers.

        Parameters
        ----------
        T : float, optional
            Temperature in Kelvin, by default from system temp
        m : int, optional
            Tail lenght in number of carbon atoms, by default from system

        Returns
        -------
        np.array
            array of self.free_energy_minimas. Gives the minimal free
            energy at every aggregation number (starting from one in the
            array)
        """
        # n is the maximal aggregation number
        n = 1000
        if T is not None or m is not None:
            # It's either updated or the same...
            self.micelles = self.get_micelles(
                T, m, **{k: True for k in self.wanted_keys}
            )

        sizes = np.arange(1, n)
        chempots = np.zeros((n - 1, len(self.wanted_keys)))

        # Loop over sizes and micelle types, if it's optimisable do it.
        # If optimised geometry is not feasible give arbitrary high value.
        for i, size in enumerate(sizes):
            chempots[i, :] = self.get_chempots(size)

        # Get the minima
        self.free_energy_minimas = np.apply_along_axis(min, axis=1, arr=chempots)

        # Get the types
        chempots_and_minima = np.concatenate(
            (chempots, np.reshape(self.free_energy_minimas, (-1, 1))), axis=1
        )

        self.free_energy_types = np.apply_along_axis(
            lambda x: np.where(np.isclose(x[:-1], x[-1]))[0][0],
            axis=1,
            arr=chempots_and_minima,
        )

        self.free_energy_minimas[0] = 0.0

        self.free_energy_types = [self.wanted_keys[i] for i in self.free_energy_types]

        return self.free_energy_minimas

    def get_micelles(
        self, T, m, spheres=False, globular=False, rodlike=False, vesicles=False, *args
    ):
        """
        Set the micelle attribute to include all the micelles we set to True.

        Parameters
        ----------
        T : float
            temperature of system.
        m : int
            Number of carbon beads in tail.
        spheres : bool, optional
            Whether or not to take spherical micelles into account,
            by default True
        globular : bool, optional
            Whether or not to take globular micelles into account,
            by default False
        rodlike : bool, optional
            Whether or not to take rodlike micelles into account,
            by default True
        vesicles : bool, optional
            Whether or not to take bilayer vesicles into account,
            by default True

        Returns
        -------
        list of MTM.micelle micelles objects
            only the ones which were set to True
        """
        # g_0 effectively will not be used.
        g_0 = 30
        # Get all the keys we put 'true' for and have a class in types in
        wanted_types = {
            k: v
            for k, v in locals().items()
            if k in MTSystem.types.keys() and v is True
        }
        wanted_keys = [k for k in wanted_types.keys()]

        micelles = []
        for key in wanted_keys:
            micelles.append(MTSystem.types.get(key)(g_0, T, m, throw_errors=False))

        self.wanted_keys = wanted_keys
        self.micelles = micelles
        return self.micelles

    def get_chempots(self, size):
        """
        Get all micelle chempot differences for a given size.

        Parameters
        ----------
        micelles : list of micelletypes
        size : float
            micelle aggregation size.
        """
        chempots = np.zeros((len(self.micelles)))
        for i, micelle in enumerate(self.micelles):
            micelle.surfactants_number = size
            if hasattr(micelle, "optimise_radii"):
                micelle.optimise_radii(hot_start=False)
            if micelle.geometry_check or micelle.geometry_check is None:
                chempots[i] = micelle.get_delta_chempot()
            else:
                chempots[i] = 101
        return chempots

    def get_monomer_concentration(self, surfactant_concentration=None, *args):
        if surfactant_concentration is None:
            surfactant_concentration = self.surfactant_concentration
        elif surfactant_concentration is None and self.surfactant_concentration is None:
            raise ValueError(
                "Please supply surfactant concentration upon system creation or when calling this method"
            )
        else:
            self.surfactant_concentration = surfactant_concentration

        if self.free_energy_minimas is None:
            _ = self.get_free_energy_minimas(*args)

        # x_0 = newton(objective, 0.01, args=(surfactant_concentration,))

        # Seems to be a starting value issue..
        roots = root_scalar(
            self.mass_balance_objective,
            args=(surfactant_concentration,),
            bracket=(1e0, 1e-10),
            method="brentq",
        )

        if roots.converged is True:
            self.monomer_concentration = roots.root
        else:
            raise RuntimeError(
                "Error message from root finding: {:s}".format(roots.flag)
            )

        return self.monomer_concentration

    def mass_balance_objective(self, monomer_conc, surfactant_concentration):
        X_acc = self.mass_balance(monomer_conc)
        return 1.0 - surfactant_concentration / X_acc

    def mass_balance(self, monomer_conc):
        X_acc = sum(
            map(
                lambda x: (x[0] + 1) * x[1],
                enumerate(self.get_aggregate_distribution(monomer_conc)),
            )
        )
        return X_acc

    def get_aggregate_distribution(self, monomer_conc=None):
        if monomer_conc is None:
            monomer_conc = self.get_monomer_concentration()
        self.aggregate_distribution = np.zeros(self.free_energy_minimas.shape)
        for i, mu_min in enumerate(self.free_energy_minimas):
            g = i + 1
            this_value = self.get_aggregate_concentration(g, monomer_conc, mu_min)
            self.aggregate_distribution[i] = this_value
        return self.aggregate_distribution

    def get_aggregate_concentration(self, g, monomer_conc=None, mu_min=None):
        if monomer_conc is None:
            monomer_conc = self.monomer_concentration
        if mu_min is None and g > 1:
            mu_min_direct = min(self.get_chempots(g))
            # Check against linear interpolation in case there is an initial
            # value problem.
            x0 = int(np.floor(g))
            f0, f1 = self.free_energy_minimas[x0 - 1], self.free_energy_minimas[x0]
            mu_min_interpol = f0 + (g - x0) * (f1 - f0)
            if mu_min_direct < f0 and mu_min_direct > f1:
                mu_min = mu_min_direct
            else:
                mu_min = mu_min_interpol
        elif mu_min is None and g <= 1:
            mu_min = 0.0

        this_value = np.exp(g * (1.0 + np.log(monomer_conc) - mu_min) - 1.0)
        return this_value


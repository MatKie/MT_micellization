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
        self, T=298.15, m=8, spheres=True, globular=False, rodlike=True, vesicles=True
    ):
        self.free_energy_minimas = None
        self.free_energy_types = None
        self.wanted_keys = None  # set by next function
        self.micelles = self.get_micelles(
            T, m, spheres=spheres, globular=globular, rodlike=rodlike, vesicles=vesicles
        )
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
        n = 400
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

    def get_monomer_concentration(self, surfactant_conc, *args):

        if self.free_energy_minimas is None:
            free_energy_minimas = self.get_free_energy_minimas(*args)
        else:
            free_energy_minimas = self.free_energy_minimas

        free_energy_minimas[0] = 0.0

        def objective(monomer_conc, surfactant_conc):
            X_acc = sum(self.get_aggregate_distribution(monomer_conc))
            return surfactant_conc - X_acc

        # x_0 = newton(objective, 0.01, args=(surfactant_conc,))

        # Seems to be a starting value issue..
        roots = root_scalar(
            objective, args=(surfactant_conc,), bracket=(1e-3, 2e-6), method="brentq"
        )

        self.monomer_concentration = roots

        _ = self.get_aggregate_distribution

    def get_aggregate_distribution(self, monomer_conc):
        self.aggregate_distribution = np.zeros(self.free_energy_minimas.shape)
        for i, mu_min in enumerate(self.free_energy_minimas):
            g = i + 1
            this_value = g * np.exp(g * (1.0 + np.log(monomer_conc) - mu_min) - 1.0)
            self.aggregate_distribution[i] = this_value
        return self.aggregate_distribution

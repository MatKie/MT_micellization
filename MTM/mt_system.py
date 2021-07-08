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

    def __init__(self):
        self.free_energy_minimas = None
        self.free_energy_types = None

    def get_free_energy_minimas(
        self,
        T=298.15,
        m=8,
        spheres=True,
        globular=False,
        rodlike=True,
        vesicles=True,
        **kwargs
    ):
        """
        Calculate free energy of various micelle shapes over all
        aggregatin numbers.

        Parameters
        ----------
        T : float, optional
            Temperature in Kelving, by default 298.15
        m : int, optional
            Tail lenght in number of carbon atoms, by default 8
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
        np.array
            array of self.free_energy_minimas. Gives the minimal free
            energy at every aggregation number (starting from one in the
            array)
        """
        types = {
            "spheres": SphericalMicelle,
            "globular": GlobularMicelle,
            "rodlike": RodlikeMicelle,
            "vesicles": BilayerVesicle,
        }
        # n is the maximal aggregation number
        # g_0 effectively will not be used.
        g_0 = 30
        n = 400

        # Get all the keys we put 'true' for and have a class in types in
        wanted_types = {
            k: v for k, v in locals().items() if k in types.keys() and v is True
        }
        wanted_keys = [k for k in wanted_types.keys()]

        sizes = np.arange(1, n)
        chempots = np.zeros((n - 1, len(wanted_keys)))

        # Loop over micelle types and sizes, if it's optimisable do it.
        # If optimised geometry is not feasible give arbitrary high value.
        for i, key in enumerate(wanted_keys):
            micelle = types.get(key)(g_0, T, m, throw_errors=False)
            for ii, size in enumerate(sizes):
                micelle.surfactants_number = size
                if hasattr(micelle, "optimise_radii"):
                    micelle.optimise_radii(hot_start=False)
                if micelle.geometry_check or micelle.geometry_check is None:
                    chempots[ii, i] = micelle.get_delta_chempot()
                else:
                    chempots[ii, i] = 101

        # Get the minima
        self.free_energy_minimas = np.apply_along_axis(min, axis=1, arr=chempots)
        chempots_and_minima = np.concatenate(
            (chempots, np.reshape(self.free_energy_minimas, (-1, 1))), axis=1
        )

        # Get the types
        self.free_energy_types = np.apply_along_axis(
            lambda x: np.where(np.isclose(x[:-1], x[-1]))[0][0],
            axis=1,
            arr=chempots_and_minima,
        )

        self.free_energy_types = [wanted_keys[i] for i in self.free_energy_types]

        return self.free_energy_minimas

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

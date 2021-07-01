from .micelles.spherical_micelle import SphericalMicelle
from .micelles.rodlike_micelle import RodlikeMicelle
from .micelles.globular_micelle import GlobularMicelle
from .micelles.bilayer_vesicle import BilayerVesicle
import numpy as np


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
        types = {
            "spheres": SphericalMicelle,
            "globular": GlobularMicelle,
            "rodlike": RodlikeMicelle,
            "vesicles": BilayerVesicle,
        }
        g_0 = kwargs.get("g_0", 30)

        wanted_types = {
            k: v for k, v in locals().items() if k in types.keys() and v is True
        }
        wanted_keys = [k for k in wanted_types.keys()]

        n = 400
        sizes = np.arange(1, n)
        chempots = np.zeros((n - 1, len(wanted_keys)))

        for i, key in enumerate(wanted_keys):
            micelle = types.get(key)(g_0, T, m, throw_errors=False)
            for ii, size in enumerate(sizes):
                micelle.surfactants_number = size
                if hasattr(micelle, "optimise_radii"):
                    micelle.optimise_radii(hot_start=False)
                chempots[ii, i] = micelle.get_delta_chempot()

        # Get the minima
        self.free_energy_minimas = chempots.min(axis=0)
        self.free_energy_types = np.where(chempots == self.free_energy_minimas)

        print("hello")

        return True

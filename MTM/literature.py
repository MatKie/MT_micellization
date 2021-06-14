from os import read
import numpy as np
import os


class LiteratureData(object):
    """
    Data used to compare our results again taken from:

    Enders: 
    S.Enders and D.Haentschel: Fluid Phase Equilibria 153 1998 1–21 Z.
                               Thermodynamics
    Reinhardt:
    Reinhardt et al.: J. Chem. Eng. Data 2020, 65, 5897−5908.

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __init__(self):
        datapath = os.path.join(os.path.dirname(__file__), "..", "data")
        # Reinhardt Fig 2, solid line.
        self.transfer_free_energy_MT = self.read_data(
            os.path.join(datapath, "transfer_free_energy_MT.csv")
        )
        # Reinhardt Fig 2, dashed line.
        self.transfer_free_energy_SAFT = self.read_data(
            os.path.join(datapath, "transfer_free_energy_SAFT.csv")
        )
        # Reinhardt Fig 4a, solid black line.
        self.interface_sph_298_MT = self.read_data(
            os.path.join(datapath, "interface_sph_298_MT.csv")
        )
        # Reinhardt Fig 4a, solid grey line.
        self.interface_sph_330_MT = self.read_data(
            os.path.join(datapath, "interface_sph_330_MT.csv")
        )
        # Reinhardt Fig 3b, solid black line.
        self.tension_dodecane_MT = self.read_data(
            os.path.join(datapath, "tension_dodecane_MT.csv")
        )
        # Reinhardt Fig 5, solid black line.
        self.delta_mu_spherical = self.read_data(
            os.path.join(datapath, "spherical_free_energy_MT.csv")
        )
        # As above extended range
        self.delta_mu_spherical_full = self.read_data(
            os.path.join(datapath, "spherical_free_energy_full_MT.csv")
        )
        # Enders Fig 2, dotted line
        self.delta_mu_rodlike_full = self.read_data(
            os.path.join(datapath, "rodlike_free_energy_C10_MT.csv")
        )
        # Enders Fig 2, solid line
        self.delta_mu_spherical_C10 = self.read_data(
            os.path.join(datapath, "spherical_free_energy_C10_MT.csv")
        )
        # As before, dashed lines
        self.delta_mu_globular = self.read_data(
            os.path.join(datapath, "globular_free_energy_C10_298.csv")
        )

    @staticmethod
    def read_data(path):
        return np.loadtxt(path, delimiter=",")

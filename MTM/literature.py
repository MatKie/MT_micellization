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

    @staticmethod
    def read_data(path):
        return np.loadtxt(path, delimiter=",")

from os import read
import numpy as np
import os


class LiteratureData(object):
    def __init__(self):
        datapath = os.path.join(os.path.dirname(__file__), "..", "data")
        self.transfer_free_energy_MT = self.read_data(
            os.path.join(datapath, "transfer_free_energy_MT.csv")
        )
        self.transfer_free_energy_SAFT = self.read_data(
            os.path.join(datapath, "transfer_free_energy_SAFT.csv")
        )
        self.interface_sph_298_MT = self.read_data(
            os.path.join(datapath, "interface_sph_298_MT.csv")
        )
        self.interface_sph_330_MT = self.read_data(
            os.path.join(datapath, "interface_sph_330_MT.csv")
        )
        self.tension_dodecane_MT = self.read_data(
            os.path.join(datapath, "tension_dodecane_MT.csv")
        )

    @staticmethod
    def read_data(path):
        return np.loadtxt(path, delimiter=",")

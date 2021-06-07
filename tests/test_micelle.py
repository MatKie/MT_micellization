import pytest
import sys
import numpy as np
import os
from mkutils import create_fig, save_to_file
from MTM import micelle
from MTM import literature

sys.path.append("../")
this_path = os.path.dirname(__file__)
from MTM import micelle
from MTM import literature


class TestBaseMicelle:
    def test_change_g(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        assert mic.g == 10
        mic.g = 100
        assert mic.g == 100

    def test_change_T(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        assert mic.T == 298.15
        mic.T = 100
        assert mic.T == 100

    def test_change_nt(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        assert mic.nt == 12
        mic.T = 8
        assert mic.T == 8

    def test_neg_T(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        with pytest.raises(ValueError):
            mic.T = -10

    def test_neg_nt(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        with pytest.raises(ValueError):
            mic.nt = -10

    def test_neg_g(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        with pytest.raises(ValueError):
            mic.g = -10

    def test_change_g_change_Volume_length(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        V = mic.volume
        l = mic.length
        mic.g = 100
        assert V == mic.volume
        assert l == mic.length

    def test_change_nt_change_Volume_length(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        V = mic.volume
        l = mic.length
        mic.nt = 8
        assert V > mic.volume
        assert l > mic.length

    def test_change_T_change_Volume_length(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        V = mic.volume
        l = mic.length
        mic.T = 300
        assert V < mic.volume
        assert l == mic.length

    def test_change_V(self):
        mic = micelle.BaseMicelle(10, 298.15, 10)
        with pytest.raises(AttributeError, match="can't set attribute"):
            mic.volume = 10

    def test_change_length(self):
        mic = micelle.BaseMicelle(10, 298.15, 10)
        with pytest.raises(AttributeError, match="can't set attribute"):
            mic.length = 10


class TestBaseMicelleGetTransferFreeEnergy:
    def test_regress_transfer_free_energy(self):
        lit = literature.LiteratureData()
        pub_values = lit.transfer_free_energy_MT
        mic = micelle.BaseMicelle(10, 298.15, 8)
        for T, value in pub_values:
            mic.T = T
            assert value == pytest.approx(mic.get_transfer_free_energy(), 5e-3)


class TestSphericalMicelleInterface:
    def test_regress_tension_dodecane(self):
        lit = literature.LiteratureData()
        pub_values = lit.tension_dodecane_MT
        mic = micelle.SphericalMicelle(1e4, 298.15, 10)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, Temp in enumerate(pub_values[:, 0]):
            mic.T = Temp
            sigma_agg = mic._sigma_agg() * 1.38064852 * 0.01 * mic.T
            calc_values[i, 0] = Temp
            calc_values[i, 1] = sigma_agg

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_tension_dodecane"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, 1e-3)

    def test_regress_interface_free_energy_298(self):
        lit = literature.LiteratureData()
        pub_values = lit.interface_sph_298_MT
        mic = micelle.SphericalMicelle(1, 298.15, 8)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic.g = g
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_interface_free_energy()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_interface_free_energy_298"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.01)

    def test_regress_interface_free_energy_330(self):
        lit = literature.LiteratureData()
        pub_values = lit.interface_sph_330_MT
        mic = micelle.SphericalMicelle(1, 330, 8)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic.g = g
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_interface_free_energy()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_interface_free_energy_330"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.1)

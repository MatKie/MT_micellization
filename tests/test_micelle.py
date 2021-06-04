import pytest
import sys
from MTM import micelle
from MTM import literature

sys.path.append("../")

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


class TestSphericalMicelle:
    pass

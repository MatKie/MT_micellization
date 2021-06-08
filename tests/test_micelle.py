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
        assert mic.surfactants_number == 10
        mic.surfactants_number = 100
        assert mic.surfactants_number == 100

    def test_change_T(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        assert mic.temperature == 298.15
        mic.temperature = 100
        assert mic.temperature == 100

    def test_change_nt(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        assert mic.tail_carbons == 12
        mic.temperature = 8
        assert mic.temperature == 8

    def test_neg_T(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        with pytest.raises(ValueError):
            mic.temperature = -10

    def test_neg_nt(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        with pytest.raises(ValueError):
            mic.tail_carbons = -10

    def test_neg_g(self):
        mic = micelle.BaseMicelle(10, 298.15, 12)
        with pytest.raises(ValueError):
            mic.surfactants_number = -10

    def test_change_g_change_Volume_length(self):
        """
        changing aggregation size should not influence volume/length.
        """
        mic = micelle.BaseMicelle(10, 298.15, 12)
        V = mic.volume
        l = mic.length
        mic.surfactants_number = 100
        assert V == mic.volume
        assert l == mic.length

    def test_change_nt_change_Volume_length(self):
        """
        Changing # carbons in tail should enlarge length and volume.
        """
        mic = micelle.BaseMicelle(10, 298.15, 12)
        V = mic.volume
        l = mic.length
        mic.tail_carbons = 8
        assert V > mic.volume
        assert l > mic.length

    def test_change_T_change_Volume_length(self):
        """
        Change of temperature should enlarge volume but not length.
        """
        mic = micelle.BaseMicelle(10, 298.15, 12)
        V = mic.volume
        l = mic.length
        mic.temperature = 300
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

    def test_change_headgroup_area(self):
        mic = micelle.BaseMicelle(10, 298.15, 10)
        mic.headgroup_area = 0.6
        assert mic.headgroup_area == 0.6
    
    def test_change_headgroup_area_warning(self):
        mic = micelle.BaseMicelle(10, 298.15, 10)
        with pytest.warns(UserWarning):
            mic.headgroup_area = 1.5
    
    def test_change_headgroup_area_error(self):
        mic = micelle.BaseMicelle(10, 298.15, 10)
        with pytest.raises(ValueError):
            mic.headgroup_area = -2


class TestBaseMicelleGetTransferFreeEnergy:
    def test_not_implemted_error(self):
        """
        Check if we throw the right errors
        """
        mic = micelle.BaseMicelle(10, 298.15, 10)
        with pytest.raises(NotImplementedError):
            mic.get_transfer_free_energy(method="InappropriateWord")

    def test_regress_transfer_free_energy(self):
        """
        Compare values to extracted values from Reinhardt et al. (see
        literature module.). Data for a 8 carbon tail at 298.15.
        """
        lit = literature.LiteratureData()
        pub_values = lit.transfer_free_energy_MT
        mic = micelle.BaseMicelle(10, 298.15, 8)
        for T, value in pub_values:
            mic.temperature = T
            calculated = mic.get_transfer_free_energy()
            assert calculated == pytest.approx(value, 5e-3)


class TestSphericalMicelleInterfaceFreeEnergy:
    def test_regress_tension_dodecane(self):
        """
        Compare values to extracted values from Reinhardt et al.
        interfacial tension of n-dodecane - water at room temp. 
        """
        lit = literature.LiteratureData()
        pub_values = lit.tension_dodecane_MT
        mic = micelle.SphericalMicelle(1e4, 298.15, 10)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, Temp in enumerate(pub_values[:, 0]):
            mic.temperature = Temp
            sigma_agg = mic._sigma_agg() * 1.38064852 * 0.01 * mic.temperature
            calc_values[i, 0] = Temp
            calc_values[i, 1] = sigma_agg

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_tension_dodecane"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, 1e-3)

    def test_regress_interface_free_energy_298(self):
        """
        Compare values to extracted values from Reinhardt et al.
        Interfacial free energy of an octyl tail at room temp over 
        various agg. sizes. 
        """
        lit = literature.LiteratureData()
        pub_values = lit.interface_sph_298_MT
        mic = micelle.SphericalMicelle(1, 298.15, 8)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic.surfactants_number = g
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_interface_free_energy()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_interface_free_energy_298"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.12)

    def test_regress_interface_free_energy_330(self):
        """
        Compare values to extracted values from Reinhardt et al.
        Interfacial free energy of an octyl tail at 330 K over 
        various agg. sizes. 
        """
        lit = literature.LiteratureData()
        pub_values = lit.interface_sph_330_MT
        mic = micelle.SphericalMicelle(1, 330, 8)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic.surfactants_number = g
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_interface_free_energy()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_interface_free_energy_330"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.1)

    def test_not_implemted_error_curv(self):
        """
        Check if we throw the right errors
        """
        mic = micelle.SphericalMicelle(10, 298.15, 10)
        with pytest.raises(NotImplementedError):
            mic.get_interface_free_energy(curvature="InappropriateWord")

    def test_not_implemted_error_method(self):
        """
        Check if we throw the right errors
        """
        mic = micelle.SphericalMicelle(10, 298.15, 10)
        with pytest.raises(NotImplementedError):
            mic.get_interface_free_energy(method="InappropriateWord")

class TestSphericalMicelleVDWStericFreeEnergy():
    def test_not_implemented_error_method(self):
        """
        Check if we throw the right errors
        """
        mic = micelle.SphericalMicelle(10, 298.15, 10)
        with pytest.raises(NotImplementedError):
            mic.get_steric_free_energy(method="InappropriateWord")
    
    def test_valuer_error(self):
        '''
        Check if we throw an error when headgroup area is too large
        '''
        mic = micelle.SphericalMicelle(5, 298, 3, 0.99)
        with pytest.raises(ValueError):
            _ = mic.get_steric_free_energy()

    def test_positive_value(self):
        '''
        Check if we get a positive value
        '''
        mic = micelle.SphericalMicelle(10, 298.15, 10, 0.49)
        calculated = mic.get_steric_free_energy()
        assert calculated > 0

class TestSphericalMicelleNagarajanDeformationFreeEnergy():
    def test_not_implemented_error_method(self):
        """
        Check if we throw the right errors
        """
        mic = micelle.SphericalMicelle(10, 298.15, 10)
        with pytest.raises(NotImplementedError):
            mic.get_deformation_free_energy(method="InappropriateWord")

    def test_positive_value(self):
        mic = micelle.SphericalMicelle(10, 298.15, 10, 0.49)
        calculated = mic.get_deformation_free_energy()
        assert calculated > 0

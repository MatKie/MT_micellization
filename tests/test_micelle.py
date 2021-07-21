from MTM.micelles.rodlike_micelle import RodlikeMicelle
from MTM.micelles._rodlike_derivative import RodlikeMicelleDerivative
from numpy.core.numeric import roll
import pytest
import sys
import numpy as np
import os
from mkutils import create_fig, save_to_file
from MTM import literature
import MTM.micelles as micelle

sys.path.append("../")
this_path = os.path.dirname(__file__)


def calc_mean_of_deviation(calc, pub):
    n = len(calc)
    if n != len(pub):
        raise RuntimeError("Vectors to be compared of different lenght!")

    diff = calc / pub - np.ones(pub.shape)
    mean = np.mean(diff)

    return np.abs(mean)


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


class TestSphericalMicelleVDWStericFreeEnergy:
    def test_not_implemented_error_method(self):
        """
        Check if we throw the right errors
        """
        mic = micelle.SphericalMicelle(10, 298.15, 10)
        with pytest.raises(NotImplementedError):
            mic.get_steric_free_energy(method="InappropriateWord")

    def test_valuer_error(self):
        """
        Check if we throw an error when headgroup area is too large
        """
        mic = micelle.SphericalMicelle(5, 298, 3, 0.99)
        with pytest.raises(ValueError):
            _ = mic.get_steric_free_energy()

    def test_positive_value(self):
        """
        Check if we get a positive value
        """
        mic = micelle.SphericalMicelle(10, 298.15, 10, 0.49)
        calculated = mic.get_steric_free_energy()
        assert calculated > 0


class TestSphericalMicelleNagarajanDeformationFreeEnergy:
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


class TestSphericalMicelleDeltaMu:
    def test_regress_spherical_narrow(self):
        """
        Compare values to extracted values from Reinhardt et al.
        delta_mu for spherical micelles at 298.15 K.
        Aggregation sizes betwee 20 and 45
        """
        lit = literature.LiteratureData()
        pub_values = lit.delta_mu_spherical
        mic = micelle.SphericalMicelle(1, 298.15, 8)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic.surfactants_number = g
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_delta_chempot()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_delta_mu"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.0075)

    def test_regress_spherical_wide(self):
        """
        Compare values to extracted values from Reinhardt et al.
        delta_mu for spherical micelles at 298.15 K.
        Aggregation sizes for all given in lit.
        """
        lit = literature.LiteratureData()
        pub_values = lit.delta_mu_spherical_full
        mic = micelle.SphericalMicelle(1, 298.15, 8)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic.surfactants_number = g
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_delta_chempot()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_delta_mu"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.075)

    def test_regress_spherical_C10(self):
        """
        Compare values to extracted values from Reinhardt et al.
        delta_mu for spherical micelles at 298.15 K.
        Aggregation sizes for all given in lit.
        """
        lit = literature.LiteratureData()
        pub_values = lit.delta_mu_spherical_C10
        mic = micelle.SphericalMicelle(1, 298.00, 10)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic.surfactants_number = g
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_delta_chempot()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_delta_mu_sph_C10"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.1)

        mean = calc_mean_of_deviation(calc_values[:, 1], pub_values[:, 1])

        assert mean < 0.005


class TestRodlikeMicelle:
    def test_cap_height(self):
        """
        Test if this is the same than 'h' here:
        http://www.ambrsoft.com/TrigoCalc/Sphere/Cap/SphereCap.htm
        """
        mic = micelle.RodlikeMicelle(10, 298.15, 10)
        rs = 2.0
        rc = 1.5
        mic._r_sph = rs
        mic._r_cyl = rc
        assert mic.cap_height == pytest.approx(0.68, abs=1e-2)

    def test_cap_volume(self):
        mic = micelle.RodlikeMicelle(110, 298.15, 8)
        rs = 1.0
        rc = 0.9
        mic._r_sph = rs
        mic._r_cyl = rc
        volume = 8.0 / 3.0 * np.pi * rs ** 3 - 1.62
        calc = mic.surfactants_number_cap * mic.volume
        assert calc == pytest.approx(volume, abs=1e-2)

    def test_cap_area(self):
        mic = micelle.RodlikeMicelle(110, 298.15, 8)
        rs = 1.0
        rc = 0.9
        mic._r_sph = rs
        mic._r_cyl = rc
        area = 8.0 * np.pi * rs ** 2 - 7.08
        calc = mic.area_per_surfactant_cap * mic.surfactants_number_cap
        assert calc == pytest.approx(area, abs=1e-2)

    def test_sum_g(self):
        for g in range(1, 300):
            mic = micelle.RodlikeMicelle(g, 298.15, 8)
            g_cap = mic.surfactants_number_cap
            g_cyl = mic.surfactants_number_cyl
            assert (g_cap + g_cyl) == pytest.approx(g)


class TestRodlikeMicelleVDWStericFreeEnergy:
    def test_positive_value(self):
        """
        Check if we get a positive value
        """
        mic = micelle.RodlikeMicelle(100, 298.15, 10, 0.49)
        calculated = mic.get_steric_free_energy()
        assert calculated > 0


class TestRodlikeMicelleNagarajanDeformationFreeEnergy:
    def test_positive_value(self):
        mic = micelle.RodlikeMicelle(10, 298.15, 10, 0.49)
        calculated = mic.get_deformation_free_energy()
        assert calculated > 0


class TestRodlikeMicelleOptimiseRadii:
    def test_optimise_micelle(self):
        mic = micelle.RodlikeMicelle(125, 298.15, 10, throw_errors=False)
        rc, rs = mic.radius_cylinder, mic.radius_sphere
        mic.optimise_radii()
        assert mic.radius_cylinder < mic.radius_sphere
        assert rc != mic.radius_cylinder
        assert rs != mic.radius_sphere


class TestRodlikeMicelleFullFreeEnergy:
    def test_regress_rodlike(self):
        """
        Compare values to extracted values from Enders.
        delta_mu for rodlike micelles at 298.15 K.
        Aggregation sizes for all given in lit.
        """
        lit = literature.LiteratureData()
        pub_values = lit.delta_mu_rodlike_full
        mic = micelle.RodlikeMicelle.optimised_radii(125, 298, 10, throw_errors=False)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic.surfactants_number = g
            mic.optimise_radii(x_0=[1.3, 1.0])
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_delta_chempot()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.set_ylim(min(pub_values[:, 1]) - 1, max(pub_values[:, 1]) + 2)
        ax.legend()

        save_to_file(os.path.join(this_path, "regress_delta_mu_rod"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.05)

        mean = calc_mean_of_deviation(calc_values[:, 1], pub_values[:, 1])

        assert mean < 0.005

    def test_unsteadiness(self):
        """
        There is/was an unsteadiness in the chemical potential difference
        over aggregation number. This test benchmarks if we can get rid
        of that.
        """
        mic = micelle.RodlikeMicelle.optimised_radii(80, 298, 10, throw_errors=False)
        gs = np.linspace(80, 180, 101)
        calc_values = np.zeros(gs.shape)

        def run_and_assert(x_0):
            for i, g in enumerate(gs):
                mic.surfactants_number = g
                mic.optimise_radii(x_0=x_0, hot_start=False)
                calc_values[i] = mic.get_delta_chempot()

            for i, i_plus_1 in zip(calc_values, calc_values[1:]):
                try:
                    assert i_plus_1 - i < 1e-8
                except AssertionError:
                    lit = literature.LiteratureData()
                    pub_values = lit.delta_mu_rodlike_full
                    fig, ax = create_fig(1, 1)
                    ax = ax[0]
                    ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
                    ax.plot(gs, calc_values, label="calc")

                    ax.set_ylim(min(pub_values[:, 1]) - 1, max(pub_values[:, 1]) + 2)
                    ax.legend()

                    save_to_file(
                        os.path.join(
                            this_path,
                            "regress_delta_mu_rod_{:.1f}_{:.1f}".format(*x_0),
                        )
                    )
                    raise AssertionError(
                        f"Unsteadiness with starting values \
                        {x_0}: {i_plus_1} - {i}"
                    )

        run_and_assert([1.3, 1.0])
        run_and_assert([1.2, 1.0])
        run_and_assert([1.0, 0.8])


class TestGlobularMicelleFullFreeEnergy:
    """
    Compare values to extracted values from Enders.
    delta_mu for rodlike micelles at 298.15 K.
    Aggregation sizes for all given in lit.
    """

    def test_regress_globular(self):
        lit = literature.LiteratureData()
        pub_values = lit.delta_mu_globular
        mic = micelle.GlobularMicelle(40, 298, 10)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic._g = g
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_delta_chempot()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_delta_mu_glob"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.2)


class TestBilayerVesicle:
    def test_bilayer_vesicle_sanity(self):
        """
        Check if the values are somewhat in the right direction after
        optimising the radii.
        """
        mic = micelle.BilayerVesicle(100, 298.15, 10)
        mic.optimise_radii()
        mu = mic.get_delta_chempot()

        assert mu > -15.0
        assert mu < -5

    def test_negative_geometries(self):
        """
        Start with a value set which gives wrong optimised geometries
        (so far).
        """
        g = 42.734856208490356
        mic = micelle.BilayerVesicle.optimised_radii(g, 298, 10)
        assert mic.radius_inner > 0

    def test_regress_bilayer_vesicle(self):
        """
        Compare values to extracted values from Enders.
        delta_mu for bilayer vesicles micelles at 298 K.
        Aggregation sizes for all given in lit.
        """
        lit = literature.LiteratureData()
        pub_values = lit.delta_mu_bilayer_vesicle
        mic = micelle.BilayerVesicle.optimised_radii(
            pub_values[0, 0], 298, 10, throw_errors=False
        )
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic._g = g
            mic.optimise_radii(hot_start=False)
            calc_values[i, 0] = g
            chempot = mic.get_delta_chempot()
            calc_values[i, 1] = chempot

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_delta_mu_bil_ves"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.05)

        mean = calc_mean_of_deviation(calc_values[:, 1], pub_values[:, 1])

        assert mean < 0.005


class TestBilayerVesicleInterface:
    def test_regress_bilayer_vesicle_298(self):
        """
        Compare values to extracted values from Reinhardt et al.
        interface free energy for bilayer vesicles micelles at 298 K.
        Aggregation sizes for all given in lit.
        """
        lit = literature.LiteratureData()
        pub_values = lit.interface_bilayer_vesicle_C8
        mic = micelle.BilayerVesicle.optimised_radii(125, 298.15, 8, throw_errors=True)
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic._g = g
            mic.optimise_radii()
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_interface_free_energy()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(
            os.path.join(this_path, "regress_interface_free_energy_bil_ves_C8_298")
        )

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.01)

        mean = calc_mean_of_deviation(calc_values[:, 1], pub_values[:, 1])

        assert mean < 0.005

    def test_regress_bilayer_vesicle_330(self):
        """
        Compare values to extracted values from Reinhardt et al.
        interface free energy for bilayer vesicles micelles at 330 K.
        Aggregation sizes for all given in lit.
        """
        lit = literature.LiteratureData()
        pub_values = lit.interface_bilayer_vesicle_C8_330
        mic = micelle.BilayerVesicle.optimised_radii(
            pub_values[0, 0], 330, 8, throw_errors=False
        )
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        for i, g in enumerate(pub_values[:, 0]):
            mic._g = g
            mic.optimise_radii(hot_start=False)
            calc_values[i, 0] = g
            calc_values[i, 1] = mic.get_interface_free_energy()

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc")

        ax.legend()

        save_to_file(
            os.path.join(this_path, "regress_interface_free_energy_bil_ves_C8_330")
        )

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.01)

        mean = calc_mean_of_deviation(calc_values[:, 1], pub_values[:, 1])

        assert mean < 0.005

    def test_deriv_vs_optim(self):
        #  works from 20 to 155 only (w.o. hotstart)
        for g in range(20, 155):
            mic_1 = micelle.BilayerVesicle(g, 298.15, 8, throw_errors=False)
            mic_2 = micelle.BilayerVesicle(g, 298.15, 8, throw_errors=False)
            mic_1.optimise_radii(method="derivative")
            mic_2.optimise_radii(method="objective")

            assert mic_1.radius_outer == pytest.approx(mic_2.radius_outer)
            assert mic_2.radius_outer == pytest.approx(mic_2.radius_outer)

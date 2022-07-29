import pytest
import sys
import numpy as np
import os
from mkutils import create_fig, save_to_file
from MTM import literature
from MTM import MTSystem
import MTM.micelles as micelle

sys.path.append("../")
this_path = os.path.dirname(__file__)


class TestAverages:
    def test_number_average(self):
        lit = literature.LiteratureData()
        pub = lit.num_average_C8
        fig, ax = create_fig(2, 1)
        ax2 = ax[1]
        ax = ax[0]
        average = []
        monomer = []
        weightav = []
        MTS = MTSystem(spheres=True, rodlike=False, vesicles=False)
        MTS._bounds = (1, 150)
        x = np.logspace(-4, 0, 20)
        for xs in x:
            # Awful hack, fix this by making these things attributes!
            MTS.aggregate_distribution = None
            MTS.monomer_concentration = None
            MTS.surfactant_concentration = xs
            average.append(MTS.get_aggregation_number())
            weightav.append(MTS.get_aggregation_number(order="weight"))
            monomer.append(MTS.monomer_concentration)
            ax2.plot(MTS.sizes[1:], MTS.sizes[1:] * MTS.aggregate_distribution[1:])
        # ax.plot(x, average, lw=2, marker="o")
        ax.plot(pub[:, 0], pub[:, 1], lw=2, label="pub")
        ax.plot(monomer, average, lw=2, ls="", label="calc", marker="o")
        ax.plot(monomer, weightav, lw=2, ls="", label="calc - weight", marker="o")
        # ax.plot(x, average)

        ax.legend()
        ax.set_xlabel("$X_1 / -$")
        ax.set_ylabel("$g_n / -$")
        save_to_file(os.path.join(this_path, "number_averages"))

        for xm, gn in zip(monomer[3:], average[3:]):
            assert gn == pytest.approx(np.interp(xm, pub[:, 0], pub[:, 1]), abs=0.2)

    def test_not_implemented_error(self):
        MTS = MTSystem(vesicles=False, rodlike=False)
        MTS._bounds = (1, 150)
        with pytest.raises(NotImplementedError):
            MTS.surfactant_concentration = 0.1
            MTS.get_aggregation_number(order="orderPLEASE")


class TestGetFreeEnergyMinimas:
    def test_return(self):
        MTS = MTSystem()
        ret = MTS.get_free_energy_minimas()
        assert isinstance(ret, np.ndarray)

    def test_regress_combined_minima(self):
        lit = literature.LiteratureData()
        pub_values = lit.overall_minima_C8_298
        MTS = MTSystem()
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        calc_values = np.zeros(pub_values.shape)
        hot_start = False
        for i, g in enumerate(pub_values[:, 0]):
            calc_values[i, 0] = g
            calc_values[i, 1] = min(MTS.get_chempots(g, hot_start=hot_start))
            hot_start = False

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub", lw=2)
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc", ls="", marker="o")

        ax.set_xlabel("Micelle size")
        ax.set_ylabel("$\Delta\mu^0_g / k_b T$")
        ax.legend()

        save_to_file(os.path.join(this_path, "regress_combined_minima"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.1)

        for pub, calc in zip(pub_values[30:, 1], calc_values[30:, 1]):
            assert calc == pytest.approx(pub, abs=0.01)

    def test_plot_combined_minima(self):
        if os.path.isfile(os.path.join(this_path, "plot_combined_minima.png")):
            return True
        fig, ax = create_fig(1, 1)
        ax = ax[0]

        for i, item in enumerate([8, 10, 12]):
            MTS = MTSystem(m=item)
            minimas = MTS.get_free_energy_minimas()
            ax.plot(
                MTS.sizes, minimas, label="{:d}".format(item), color="C{:d}".format(i)
            )
            ax.plot(MTS.sizes, minimas, ls="", marker="o", color="C{:d}".format(i))

        ax.legend()
        ax.axvline(28, ls="--", color="C0", lw=2)
        ax.axvline(56, ls="--", color="C2", lw=2)
        ax.set_xlim(0, 75)

        ax.set_xlabel("Micelle size")
        ax.set_ylabel("$\Delta\mu^0_g / k_b T$")
        save_to_file(os.path.join(this_path, "plot_combined_minima"))

    def test_sph_vs_vesicle_lower(self):
        """
        At very low aggregation numbers spherical micelles _should_ be the
        smallest..
        The switch is around agg.nr. 28. Hence 27 should be spherical.
        """
        MTS = MTSystem()
        values = MTS.get_chempots(27)
        assert values[0] < values[2]

    def test_sph_vs_vesicle_upper(self):
        """
        At very low aggregation numbers spherical micelles _should_ be the
        smallest..
        The switch is around agg.nr. 28. Hence 29 should be bil.ves.
        """
        MTS = MTSystem()
        values = MTS.get_chempots(29)
        assert values[0] > values[2]

    @pytest.mark.xfail
    def test_sph_vs_vesicle_lower_m12(self):
        """
        At very low aggregation numbers spherical micelles _should_ be the
        smallest..
        The switch is around agg.nr. 56. Hence 55 should be spherical.
        """
        MTS = MTSystem(m=12)
        values = MTS.get_chempots(55)
        assert values[0] < values[2]

    def test_sph_vs_vesicle_upper_m12(self):
        """
        At very low aggregation numbers spherical micelles _should_ be the
        smallest..
        The switch is around agg.nr. 56. Hence 57 should be bil.ves.
        """
        MTS = MTSystem(m=12)
        values = MTS.get_chempots(57)
        assert values[0] > values[2]

    def test_vesicle_inflection(self):
        """
        This is a debug test to look at geometries of micelles both
        sides of the inflection point.
        Micelles only possible starting number 27-ish?
        """
        MTSSmall = MTSystem()
        MTSBig = MTSystem()

        small_chempots = MTSSmall.get_chempots(26, hot_start=True)
        big_chempots = MTSBig.get_chempots(27, hot_start=True)

        SmallMicelle = MTSSmall.micelles[2]
        BigMicelle = MTSBig.micelles[2]

        assert SmallMicelle.geometry_check == True
        assert BigMicelle.geometry_check == True
        assert small_chempots[0] < small_chempots[-1]
        assert big_chempots[0] > big_chempots[-1]


class TestGetMonomerConcentration:
    def test_return(self):
        MTS = MTSystem()
        ret = MTS.get_monomer_concentration(0.1)

    def compare(self, pub, MTS, factor=1e6):
        calc = np.zeros(pub.shape)
        MTS.get_monomer_concentration()
        for i, size in enumerate(pub[:, 0]):
            calc[i, 0] = size
            calc[i, 1] = factor * MTS.get_aggregate_concentration(size)

        for (size, pubi), calci in zip(enumerate(pub[:, 1]), calc[:, 1]):
            try:
                assert calci == pytest.approx(pubi, abs=0.6)
            except AssertionError:
                flag = False
                mssg = "Assertion error for system: Xs {:f}, C {:d} at size {:f}".format(
                    MTS.surfactant_concentration, MTS.m, calc[size, 0]
                )
                return calc, flag, mssg
        return calc, True, ""

    def test_regress_concentrations(self):
        lit = literature.LiteratureData()
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        pub = lit.Xg_C12_X_005
        MTS = MTSystem(T=298, m=12, surfactant_concentration=0.005)
        calc, flag1, mssg1 = self.compare(pub, MTS)
        color = "C0"
        ax.plot(pub[:, 0], pub[:, 1], color=color, label="X005")
        ax.plot(calc[:, 0], calc[:, 1], ls="", marker="o", color=color)

        pub = lit.Xg_C12_X_05
        MTS = MTSystem(T=298, m=12, surfactant_concentration=0.05)
        calc, flag2, mssg2 = self.compare(pub, MTS)
        color = "C1"
        ax.plot(pub[:, 0], pub[:, 1], color=color, label="X05")
        ax.plot(calc[:, 0], calc[:, 1], ls="", marker="o", color=color)

        pub = lit.Xg_C12_X_15
        MTS = MTSystem(T=298, m=12, surfactant_concentration=0.15)
        calc, flag3, mssg3 = self.compare(pub, MTS)
        color = "C3"
        ax.plot(pub[:, 0], pub[:, 1], color=color, label="X15")
        ax.plot(calc[:, 0], calc[:, 1], ls="", marker="o", color=color)

        ax.legend()

        ax.set_xlabel("Micelle size")
        ax.set_ylabel("Molfrac / mol/mol")
        save_to_file(os.path.join(this_path, "regress_concentration_distributions"))

        if not flag1 or not flag2 or not flag3:
            raise AssertionError("{:s}\n\n{:s}\n\n{:s}".format(mssg1, mssg2, mssg3))

    def test_regress_taillengths(self):
        lit = literature.LiteratureData()
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        pub = lit.Xg_C8_X_15
        MTS = MTSystem(T=298, m=8, surfactant_concentration=0.15)
        calc, flag1, mssg1 = self.compare(pub, MTS, factor=1e5)
        color = "C0"
        ax.plot(pub[:, 0], pub[:, 1], color=color, label="C8", lw=2)
        ax.plot(calc[:, 0], calc[:, 1], ls="", marker="o", color=color)
        pub = lit.Xg_C10_X_15
        MTS = MTSystem(T=298, m=10, surfactant_concentration=0.15)
        calc, flag2, mssg2 = self.compare(pub, MTS, factor=1e5)
        color = "C1"
        ax.plot(pub[:, 0], pub[:, 1], color=color, label="C10", lw=2)
        ax.plot(calc[:, 0], calc[:, 1], ls="", marker="o", color=color)
        pub = lit.Xg_C12_X_15_fig6
        MTS = MTSystem(T=298, m=12, surfactant_concentration=0.15)
        calc, flag3, mssg3 = self.compare(pub, MTS, factor=1e5)
        color = "C3"
        ax.plot(pub[:, 0], pub[:, 1], color=color, label="C12", lw=2)
        ax.plot(calc[:, 0], calc[:, 1], ls="", marker="o", color=color)

        ax.legend()
        ax.set_xlabel("Micelle size")
        ax.set_ylabel("Molfrac / mol/mol")
        save_to_file(os.path.join(this_path, "regress_taillength_distributions"))

        if not flag1 or not flag2 or not flag3:
            raise AssertionError("{:s}\n\n{:s}\n\n{:s}".format(mssg1, mssg2, mssg3))


class TestSGTSystems:
    def test_optimise_radii(self):
        # make a test of total energy of micellisation with SGT and non
        # SGT, maybe plot it
        # check if ift gets calculated multiple times
        chempot_args = {"interface_method": "sgt"}
        SGT = MTSystem(
            T=300, m=9, surfactant_concentration=0.15, chempot_args=chempot_args,
        )
        SGT_normal = MTSystem(T=300, m=9, surfactant_concentration=0.15,)
        SGT._bounds = (1, 200)
        SGT_normal._bounds = (1, 200)

        SGT.get_free_energy_minimas()
        SGT_normal.get_free_energy_minimas()

        fig, ax = create_fig(1, 1)
        ax = ax[0]

        ax.plot(SGT.sizes, SGT.free_energy_minimas)
        ax.plot(SGT_normal.sizes, SGT_normal.free_energy_minimas)

        save_to_file(os.path.join(this_path, "SGT_vs_normal"))

        for sgt_calced, normal_calced in zip(
            SGT.free_energy_minimas, SGT_normal.free_energy_minimas
        ):
            assert sgt_calced > normal_calced

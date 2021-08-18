from MTM.micelles._transfer_saft_gamma import TransferSaftGamma
from MTM.micelles._transfer_saft import TransferSaft
from MTM import MTSystem
from MTM.literature import LiteratureData
import pytest
import sys
import numpy as np
import os
from mkutils import create_fig, save_to_file
import MTM.micelles as micelle

sys.path.append("../")
this_path = os.path.dirname(__file__)
figure_path = os.path.join(this_path, "transfer_plots")


class TestTransfer(object):
    def test_set_temperature(self):
        lit = LiteratureData()
        this_lit = lit.mu_tr_alkanes
        alkanes_lit = this_lit[:7, :]
        regress_lit = this_lit[7:, :]

        fig, ax = create_fig(1, 1)
        ax = ax[0]

        Transfer = TransferSaftGamma(tail_length=8, alkyl_ends=False)
        Transfer._ch2_calc_bounds = (5, 12)
        Transfer.temperature = 298.15
        transfer_data = Transfer._mu_alkane
        c = np.polyfit(range(*Transfer._ch2_calc_bounds), transfer_data, 1)
        p = np.poly1d(c)
        cn = np.linspace(4.5, 11.5, 100)
        ax.plot(
            range(*Transfer._ch2_calc_bounds),
            transfer_data,
            color="C0",
            label="this work - full alkanes",
            marker="s",
            ls="",
        )
        ax.plot(cn, p(cn), color="C0", lw=2, ls="--")
        Transfer = TransferSaftGamma(
            tail_length=8, alkyl_ends=True, hydrate_c1_carbon=True
        )
        Transfer._ch2_calc_bounds = (5, 12)
        Transfer.temperature = 298.15
        transfer_data = Transfer._mu_alkane
        c = np.polyfit(range(*Transfer._ch2_calc_bounds), transfer_data, 1)
        p = np.poly1d(c)
        cn = np.linspace(4.5, 11.5, 100)
        ax.plot(
            range(*Transfer._ch2_calc_bounds),
            transfer_data,
            color="C2",
            label="this work - alkyl tails + hydrated",
            marker="s",
            ls="",
        )
        ax.plot(cn, p(cn), color="C2", lw=2, ls="--")
        Transfer = TransferSaftGamma(tail_length=8)
        Transfer._ch2_calc_bounds = (5, 12)
        Transfer.temperature = 298.15
        transfer_data = Transfer._mu_alkane
        c = np.polyfit(range(*Transfer._ch2_calc_bounds), transfer_data, 1)
        p = np.poly1d(c)
        cn = np.linspace(4.5, 11.5, 100)
        ax.plot(
            range(*Transfer._ch2_calc_bounds),
            transfer_data,
            color="C3",
            label="this work - alkyl tails",
            marker="s",
            ls="",
        )
        ax.plot(cn, p(cn), color="C3", lw=2, ls="--")
        ax.plot(
            alkanes_lit[:, 0],
            alkanes_lit[:, 1],
            color="k",
            marker="o",
            label="Published",
            ls="",
        )
        ax.plot(regress_lit[:, 0], regress_lit[:, 1], color="k", ls="-", lw=2)
        ax.legend()
        ax.set_xlabel("C$_n$ / -")
        ax.set_ylabel(r"$\mu_{A,Tr}/k_BT$")
        save_to_file(os.path.join(figure_path, "mu_alkanes_gamma"))

        assert Transfer.mu_ch2 == pytest.approx(-1.609, abs=0.001)

    def test_transfer_base_micelle(self):
        BM = micelle.BaseMicelle(40, 300, 8)
        BM._transfer_saft_gamma.hydrate_c1_carbon = True
        mu_tr = BM.get_transfer_free_energy(method="assoc_saft_gamma")

        assert mu_tr == pytest.approx(-14.04, abs=0.1)

    def test_transfer_monotonically_decrease(self):
        BM1 = micelle.BaseMicelle(40, 300, 6)
        BM2 = micelle.BaseMicelle(40, 300, 9)
        BM3 = micelle.BaseMicelle(40, 300, 11)

        mu_tr1 = BM1.get_transfer_free_energy(method="assoc_saft_gamma")
        mu_tr2 = BM2.get_transfer_free_energy(method="assoc_saft_gamma")
        mu_tr3 = BM3.get_transfer_free_energy(method="assoc_saft_gamma")

        assert mu_tr1 > mu_tr2 > mu_tr3

    def test_transfer_emp_saft(self):
        lit = LiteratureData()
        this_data = lit.mu_tr_C8
        temps = np.linspace(280, 360, 30)
        BM1 = micelle.BaseMicelle(40, 280, 8)
        BM2 = micelle.BaseMicelle(40, 280, 8)
        BM3 = micelle.BaseMicelle(40, 280, 8)
        saft, emp, saft_alkyl, saft_hydrated = [], [], [], []
        for temp in temps:
            BM1._transfer_saft_gamma.alkyl_ends = False
            BM1.temperature = temp
            saft.append(BM1.get_transfer_free_energy(method="assoc_saft_gamma"))
            emp.append(BM1.get_transfer_free_energy(method="empirical"))
            BM2._transfer_saft_gamma.alkyl_ends = True
            BM2.temperature = temp
            saft_alkyl.append(BM2.get_transfer_free_energy(method="assoc_saft_gamma"))
            BM3._transfer_saft_gamma.alkyl_ends = True
            BM3.temperature = temp
            BM3._transfer_saft_gamma.hydrate_c1_carbon = True
            saft_hydrated.append(
                BM3.get_transfer_free_energy(method="assoc_saft_gamma")
            )

        fig, ax = create_fig(1, 1)
        ax = ax[0]

        ax.plot(temps, saft, ls="-", color="C0", label="this work - full alkanes", lw=2)
        ax.plot(
            temps,
            saft_hydrated,
            ls="-",
            color="C2",
            label="this work - alkyl tails + hydrated",
            lw=2,
        )
        ax.plot(
            temps,
            saft_alkyl,
            ls="-",
            color="C3",
            label="this work - alkyl tails",
            lw=2,
        )
        ax.plot(temps, emp, ls="--", color="k", label="Empirical", lw=2)
        ax.plot(
            this_data[:, 0], this_data[:, 1], color="k", lw=2, label="Published SAFT"
        )

        ax.set_ylabel(r"$\mu_g$")
        ax.legend()
        save_to_file(os.path.join(figure_path, "transfer_emp_vs_saft_gamma"))


class TestAverages:
    def test_number_average(self):
        lit = LiteratureData()
        empirical = lit.num_average_C8
        published = lit.num_average_C8_transfer
        T = 298.15
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        average, average_3, average_4 = [], [], []
        monomer, monomer_3, monomer_4 = [], [], []
        average_2 = []
        monomer_2 = []
        MTS = MTSystem(T, spheres=True, rodlike=False, vesicles=False)
        MTS._bounds = (1, 150)
        chempot_args = {"transfer_method": "assoc_saft_gamma"}
        MTS2 = MTSystem(
            T, spheres=True, rodlike=False, vesicles=False, chempot_args=chempot_args
        )
        MTS2.micelles[0]._transfer_saft_gamma.alkyl_ends = False
        MTS2._bounds = (1, 150)
        MTS3 = MTSystem(
            T, spheres=True, rodlike=False, vesicles=False, chempot_args=chempot_args
        )
        MTS3._bounds = (1, 150)
        MTS3.micelles[0]._transfer_saft_gamma.alkyl_ends = True
        MTS3.micelles[0]._transfer_saft_gamma.hydrate_c1_carbon = True
        MTS4 = MTSystem(
            T, spheres=True, rodlike=False, vesicles=False, chempot_args=chempot_args
        )
        MTS4._bounds = (1, 150)
        MTS4.micelles[0]._transfer_saft_gamma.alkyl_ends = True
        MTS4.micelles[0]._transfer_saft_gamma.hydrate_c1_carbon = False
        x = np.logspace(-4, 0, 20)
        for xs in x:
            # Awful hack, fix this by making these things attributes!
            for MTSi in [MTS, MTS2, MTS3, MTS4]:
                MTSi.aggregate_distribution = None
                MTSi.monomer_concentration = None
                MTSi.surfactant_concentration = xs
            average.append(MTS.get_aggregation_number())
            monomer.append(MTS.monomer_concentration)
            average_2.append(MTS2.get_aggregation_number())
            monomer_2.append(MTS2.monomer_concentration)
            average_3.append(MTS3.get_aggregation_number())
            monomer_3.append(MTS3.monomer_concentration)
            average_4.append(MTS4.get_aggregation_number())
            monomer_4.append(MTS4.monomer_concentration)
        # ax.plot(x, average, lw=2, marker="o")
        ax.plot(monomer, average, label="Empirical", color="k", ls="", marker="o")
        ax.plot(
            monomer_2[1:],
            average_2[1:],
            label="This work - full alkane",
            color="C0",
            ls="",
            marker="o",
        )
        ax.plot(
            monomer_3[1:],
            average_3[1:],
            label="This work - alkyl tail + hydrated",
            color="C2",
            ls="",
            marker="o",
        )
        ax.plot(
            monomer_4[1:],
            average_4[1:],
            label="This work - alkyl tail",
            color="C3",
            ls="",
            marker="o",
        )
        ax.plot(
            empirical[:, 0],
            empirical[:, 1],
            lw=2,
            label="Published Empirical",
            color="k",
        )
        ax.plot(
            published[:, 0],
            published[:, 1],
            lw=2,
            label="Published Saft",
            color="k",
            ls="--",
        )
        # ax.plot(x, average)

        ax.legend(loc="center right")
        ax.set_xlabel("$X_1 / -$")
        ax.set_ylabel("$g_n / -$")
        save_to_file(os.path.join(figure_path, "number_averages_transfer_gamma"))

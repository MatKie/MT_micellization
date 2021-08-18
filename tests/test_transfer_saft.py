from numpy.lib.npyio import save
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

        Transfer = TransferSaft()
        Transfer.temperature = 298.15
        fig, ax = create_fig(1, 1)
        ax = ax[0]

        transfer_data = np.array(sorted(Transfer.mu_alkanes.items()))
        c = np.polyfit(transfer_data[:, 0], transfer_data[:, 1], 1)
        p = np.poly1d(c)
        cn = np.linspace(4.5, 11.5, 100)
        ax.plot(
            transfer_data[:, 0],
            transfer_data[:, 1],
            color="C3",
            label="this work",
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
        ax.set_xlabel("C$_n$ / -")
        ax.set_ylabel(r"$\mu_{A,Tr}/k_BT$")
        save_to_file(os.path.join(figure_path, "mu_alkanes"))

        assert Transfer.mu_ch2 == pytest.approx(-1.685, abs=0.001)

    def test_transfer_base_micelle(self):
        BM = micelle.BaseMicelle(40, 300, 8)
        mu_tr = BM.get_transfer_free_energy(method="assoc_saft")

        assert mu_tr == pytest.approx(-13.15, abs=0.1)

    def test_transfer_monotonically_decrease(self):
        BM1 = micelle.BaseMicelle(40, 300, 6)
        BM2 = micelle.BaseMicelle(40, 300, 9)
        BM3 = micelle.BaseMicelle(40, 300, 11)

        mu_tr1 = BM1.get_transfer_free_energy(method="assoc_saft")
        mu_tr2 = BM2.get_transfer_free_energy(method="assoc_saft")
        mu_tr3 = BM3.get_transfer_free_energy(method="assoc_saft")

        assert mu_tr1 > mu_tr2 > mu_tr3

    def test_transfer_emp_saft(self):
        lit = LiteratureData()
        this_data = lit.mu_tr_C8
        temps = np.linspace(280, 360, 30)
        BM = micelle.BaseMicelle(40, 280, 8)
        saft, emp = [], []
        for temp in temps:
            BM.temperature = temp
            saft.append(BM.get_transfer_free_energy(method="assoc_saft"))
            emp.append(BM.get_transfer_free_energy(method="empirical"))

        fig, ax = create_fig(1, 1)
        ax = ax[0]

        ax.plot(temps, emp, ls="--", color="k", label="empirical", lw=2)
        ax.plot(temps, saft, ls="-", color="C0", label="SAFT VR-MIE (assoc)", lw=2)
        ax.plot(
            this_data[:, 0], this_data[:, 1], color="k", lw=2, label="Published SAFT"
        )

        ax.set_ylabel(r"$\mu_g$")
        ax.legend()
        save_to_file(os.path.join(figure_path, "transfer_emp_vs_saft"))


class TestAverages:
    def test_number_average(self):
        lit = LiteratureData()
        empirical = lit.num_average_C8
        published = lit.num_average_C8_transfer
        T = 298.15
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        average = []
        monomer = []
        average_2 = []
        monomer_2 = []
        MTS = MTSystem(T, spheres=True, rodlike=False, vesicles=False)
        MTS._bounds = (1, 150)
        chempot_args = {"transfer_method": "assoc_saft"}
        MTS2 = MTSystem(
            T, spheres=True, rodlike=False, vesicles=False, chempot_args=chempot_args
        )
        MTS2._bounds = (1, 150)
        x = np.logspace(-4, 0, 20)
        for xs in x:
            # Awful hack, fix this by making these things attributes!
            MTS.aggregate_distribution = None
            MTS.monomer_concentration = None
            MTS.surfactant_concentration = xs
            MTS2.aggregate_distribution = None
            MTS2.monomer_concentration = None
            MTS2.surfactant_concentration = xs
            average.append(MTS.get_aggregation_number())
            monomer.append(MTS.monomer_concentration)
            average_2.append(MTS2.get_aggregation_number())
            monomer_2.append(MTS2.monomer_concentration)
        # ax.plot(x, average, lw=2, marker="o")
        ax.plot(monomer, average, label="Empirical", color="k", ls="", marker="o")
        ax.plot(
            monomer_2, average_2, label="Transfer Saft", color="C3", ls="", marker="o"
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
            color="C3",
            ls="--",
        )
        # ax.plot(x, average)

        ax.legend()
        ax.set_xlabel("$X_1 / -$")
        ax.set_ylabel("$g_n / -$")
        save_to_file(os.path.join(figure_path, "number_averages_transfer"))

        ax.plot(x, average, label="Empirical", color="k", ls="", marker="o")
        ax.plot(x, average_2, label="Transfer Saft", color="C3", ls="", marker="o")
        ax.set_xscale("log")
        save_to_file(os.path.join(figure_path, "test"))
        fig, ax = create_fig(1, 1)
        ax = ax[0]

        ax.plot(x, monomer, label="Empirical", color="k", ls="", marker="o")
        ax.plot(x, monomer_2, label="Transfer Saft", color="C3", ls="", marker="o")
        save_to_file(os.path.join(figure_path, "test2"))


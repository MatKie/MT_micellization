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
        for i, g in enumerate(pub_values[:, 0]):
            calc_values[i, 0] = g
            calc_values[i, 1] = min(MTS.get_chempots(g))

        ax.plot(pub_values[:, 0], pub_values[:, 1], label="pub")
        ax.plot(calc_values[:, 0], calc_values[:, 1], label="calc", ls="", marker="o")

        ax.legend()

        save_to_file(os.path.join(this_path, "regress_combined_minima"))

        for pub, calc in zip(pub_values[:, 1], calc_values[:, 1]):
            assert calc == pytest.approx(pub, abs=0.0075)

    def test_sph_vs_vesicle(self):
        """
        At very low aggregation numbers spherical micelles _should_ be the
        smallest..
        """
        MTS = MTSystem()
        values = MTS.get_chempots(27)
        assert values[0] < values[2]


class TestGetMonomerConcentration:
    def test_return(self):
        MTS = MTSystem()
        ret = MTS.get_monomer_concentration(0.1)

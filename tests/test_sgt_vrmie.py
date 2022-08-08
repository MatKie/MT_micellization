import pytest
from MTM.micelles._sigma_sgt import SigmaSGT
from MTM.micelles._sigma_sgt_gamma import SigmaSGTGamma


class TestEos:
    def test_set_eos(self):
        SGT = SigmaSGT(8)
        SGT.temperature = 298.15
        eos = SGT.eos

        assert (
            eos.beta0[0, 1]
            == (1.01385065e-05 * 298.15 * 298.15)
            - (7.31757041e-03 * 298.15)
            + 1.75636346e00
        )
        assert (
            eos.KIJ0saft[0, 1]
            == -3.76764999e-06 * 298.15 * 298.15
            + 2.88068999e-03 * 298.15
            - 4.70588225e-01
        )

    def test_lle(self):
        SGT = SigmaSGT(8)
        SGT.temperature = 298.15
        rho_l1, rho_l2 = SGT.rho_l1, SGT.rho_l2

        assert rho_l1[0] * 0.018015 == pytest.approx(1026, abs=2)


class TestSGT:
    def test_calc_ift_octane_quadratic(self):
        SGT = SigmaSGT(8)
        SGT.temperature = 294.4117

        ift = SGT.get_ift()

        assert ift == pytest.approx(51.50, abs=0.01)

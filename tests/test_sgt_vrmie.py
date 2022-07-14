import pytest
from MTM.micelles._sigma_sgt import SigmaSGT


class TestEos:
    def test_set_eos(self):
        SGT = SigmaSGT(8, bij_correlation="linear")
        SGT.temperature = 298.15
        eos = SGT.eos

        assert eos.beta0[0, 1] == -0.00140671 * 298.15 + 0.90335974
        assert (
            eos.KIJ0saft[0, 1]
            == -3.76764999e-06 * 298.15 * 298.15
            + 2.88068999e-03 * 298.15
            - 4.70588225e-01
        )

    def test_lle(self):
        SGT = SigmaSGT(8, bij_correlation="linear")
        SGT.temperature = 298.15
        rho_l1, rho_l2 = SGT.rho_l1, SGT.rho_l2

        assert rho_l1[0] * 0.018015 == pytest.approx(1026, abs=2)


class TestSGT:
    def test_calc_ift_octane_quadratic(self):
        SGT = SigmaSGT(8, bij_correlation="quadratic")
        SGT.temperature = 298.15

        ift = SGT.get_ift()

        assert ift == pytest.approx(51.83, abs=0.1)

    def test_calc_ift_octane_linear(self):
        SGT = SigmaSGT(8, bij_correlation="linear")
        SGT.temperature = 298.15

        ift = SGT.get_ift()

        assert ift == pytest.approx(51.53, abs=0.1)

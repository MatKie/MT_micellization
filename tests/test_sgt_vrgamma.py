import pytest
from MTM.micelles._sigma_sgt_gamma import SigmaSGTGamma


class TestEos:
    def test_set_eos_gamma(self):
        SGT = SigmaSGTGamma(8)
        SGT.temperature = 298.15
        eos = SGT.eos

        assert eos.beta0[0, 1] == -0.0007751 * 298.15 + 0.67819916

    def test_lle_GAMMA(self):
        SGT = SigmaSGTGamma(8)
        SGT.temperature = 298.15
        rho_l1, rho_l2 = SGT.rho_l1, SGT.rho_l2

        assert rho_l1[0] * 0.018015 == pytest.approx(1000, abs=2)


class TestSGTGamma:
    def test_no_alkyl(self):
        SGT = SigmaSGTGamma(10, alkyl_tails=False)
        SGT.temperature = 310.88235294117646

        ift = SGT.get_ift()

        assert ift == pytest.approx(50.977, abs=0.01)

    def test_alkyl(self):
        SGT = SigmaSGTGamma(10)
        SGT.temperature = 310.88235294117646

        ift = SGT.get_ift()

        assert ift != pytest.approx(50.977, abs=0.01)

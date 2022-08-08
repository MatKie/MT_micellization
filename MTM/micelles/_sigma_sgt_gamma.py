from ._sigma_sgt import SigmaSGT
from sgtpy import mixture, saftgammamie, component
from sgtpy.equilibrium import lle, lle_init
from sgtpy.sgt import sgt_mix
from numpy import poly1d, array
from numpy.linalg import LinAlgError


class SigmaSGTGamma(SigmaSGT):
    # ciis for hexane = undecane acc to
    ciis = {
        4: 3.4730220740206996e-19,
        5: 4.7211276314762078e-19,
        6: 5.8921793879625545e-19,
        7: 7.3610371846705358e-19,
        8: 9.0025536743298327e-19,
        9: 1.0885424839517982e-18,
    }
    interfacial_tension_gamma_alkyl = {}
    interfacial_tension_gamma = {}

    def __init__(self, total_carbons, alkyl_tails=True):
        self.p = 101325
        self.alkyl_tails = alkyl_tails
        self.tail_carbons = total_carbons - 2
        self.n_ch3 = 1
        self.water = component(GC={"H2O": 1}, cii=1.6152389224852537e-20, Mw=18.015)
        bij_polynomial = (-0.0007751, 0.67819916)
        self.bij_polynomial = poly1d(bij_polynomial)
        self.eos = None
        self.rho_l1, self.rho_l2 = None, None
        self._T = None
        self.itmax = 25

        # sgt_mix parameters
        self.z0 = 10
        self.rho0 = "linear"  # options are 'hyperbolic' and a previous result
        self.max_dz0 = 25

    def get_ift(self, **kwargs):
        if self.alkyl_tails:
            ift = SigmaSGTGamma.interfacial_tension_gamma_alkyl.get(
                self.tail_carbons, {}
            ).get(self.temperature)
        else:
            ift = SigmaSGTGamma.interfacial_tension_gamma.get(
                self.tail_carbons, {}
            ).get(self.temperature)

        if ift == None:
            ift = self._get_ift(**kwargs)
            update_dict = {self.tail_carbons: {self.temperature: ift}}
            if self.alkyl_tails:
                SigmaSGTGamma.interfacial_tension_gamma.update(update_dict)
            else:
                SigmaSGTGamma.interfacial_tension_gamma.update(update_dict)

        return ift

    def _set_eos(self):
        n_CH3 = 1
        incr_tail_carbons = 1
        if not self.alkyl_tails:
            n_CH3 += 1
            incr_tail_carbons -= 1

        MW = n_CH3 * 15 + (self.tail_carbons + incr_tail_carbons) * 14
        alkane = component(
            GC={"CH3": n_CH3, "CH2": self.tail_carbons + incr_tail_carbons},
            cii=SigmaSGTGamma.ciis.get(self.tail_carbons),
            Mw=MW,
        )
        mix = mixture(self.water, alkane)
        mix.saftgammamie()
        eos = saftgammamie(mix)

        bij = self.bij_polynomial(self._T)
        Bij = array([[0.0, bij], [bij, 0.0]])
        eos.beta_sgt(Bij)

        return eos

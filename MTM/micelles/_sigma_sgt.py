from ._saftvr_mie import SaftVR
from sgtpy import mixture, saftvrmie, 
from sgtpy.equilibrium import lle, lle_init
from numpy import poly1d, array


class SigmaSGT(SaftVR):
    def __init__(self, tail_carbons):
        super().__init__()
        self.p = 101325
        self.tail_carbons = tail_carbons
        self.component = self._SaftVR__components[self.tail_carbons - 5]
        kij_polynomial = self._SaftVR__components[self.tail_carbons - 5]
        self.kij_polynomial = poly1d(kij_polynomial)
        self.eos

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, T):
        if T == None or T == self._T:
            pass
        elif T != self._T:
            # setup mixture+eos with lle calc and set beta matrix
            self._T = T
            self.eos = self._set_eos()
            self.rho_l1, self.rho_l2 = self._get_lle(self.eos)

    def get_ift(self):
        # run the sgt
        return 52

    def _set_eos(self):
        mix = mixture(self._SaftVR__water, self.component)
        kij = self.kij_polynomial(self._T)
        Kij = array([[0.0, kij], [kij, 0.0]])
        mix.kij_saft(Kij)
        return saftvrmie(mix)

    def _get_lle(self, eos):
        x_wr, x_ar = [1e-8, 1.-1e-8], [1.-1e-4, 1e-4]
        x_wr, x_ar = lle_init((x_wr+x_ar)/2., self._T, self.p, eos)
        sol = lle(x_wr, x_ar, (x_wr+x_ar)/2., self._T, self.p, eos, full_output=True)
        x_wr, x_ar = sol.X
        vl_wr, vl_ar = sol.v

        return x_wr/vl_wr, x_ar/vl_ar
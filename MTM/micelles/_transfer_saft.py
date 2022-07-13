try:
    from SGTPy import saftvrmie, mixture, component
except:
    from sgtpy import saftvrmie, mixture, component
import numpy as np
from ._saftvr_mie import SaftVR


class TransferSaft(SaftVR):
    def __init__(self):
        super().__init__()
        self.mixture_eos = None  # list of saftvrmiemix objects
        # dictionary cn : mu_tr
        self.mu_alkanes = {
            5: None,
            6: None,
            7: None,
            8: None,
            9: None,
            10: None,
            11: None,
        }
        self.mu_ch2 = None  # Slope of lngamma vs cn
        self._T = None

    @property
    def temperature(self):
        return self._T

    @temperature.setter
    def temperature(self, T):
        if T == None or T == self._T:
            pass
        elif T != self._T:
            self._T = T
            self._set_eos(T)
            self._calc_mu_tr()

    def _set_eos(self, T):
        self.mixture_eos = []
        for kij_coeffs, alkane in zip(
            self._SaftVR__kij_coeffs, self._SaftVR__components
        ):
            p = np.poly1d(kij_coeffs)
            kij = p(T)
            kij_arr = np.array([[0.0, kij], [kij, 0.0]])
            mix = mixture(self._SaftVR__water, alkane)
            mix.kij_saft(kij_arr)
            self.mixture_eos.append(saftvrmie(mix))

    def _calc_mu_tr(self):
        self.mu_alkanes = {}
        for i, mix in enumerate(self.mixture_eos):
            lngamma = -1.0 * mix.get_lngamma(1.0 - 1e-10, self._T, 101325)[1]
            self.mu_alkanes.update({i + 5: lngamma})

        keys = list(self.mu_alkanes.keys())
        values = [self.mu_alkanes.get(key) for key in keys]
        m, c = np.polyfit(keys, values, 1)
        self.mu_ch2 = m

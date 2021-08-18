from SGTPy import saftgammamie, mixture, component
import numpy as np


class TransferSaftGamma(object):
    def __init__(self, tail_length, alkyl_ends=True, hydrate_c1_carbon=False):
        self.tail_length = tail_length
        self.mu_ch2 = None  # Slope of lngamma vs cn
        self.mu_alkane = None  # Transfer energy of alkane
        self._mu_alkane = None  # Transfer energy of all calc. alkanes
        self._T = None
        self.pressure = 101325
        self.alkyl_ends = alkyl_ends
        self.hydrate_c1_carbon = hydrate_c1_carbon
        self._ch2_calc_bounds = (self.tail_length - 2, self.tail_length + 3)

    @property
    def temperature(self):
        return self._T

    @temperature.setter
    def temperature(self, T):
        if T == None or T == self._T:
            pass
        elif T != self._T:
            self._T = T
            self._calc_mu_tr(T)

    def _calc_mu_tr(self, T):
        # Calc slope
        water = component(GC={"H2O": 1})
        nr_ch3 = 2
        self._mu_alkane = []
        if self.alkyl_ends:
            nr_ch3 -= 1
        reduce_ch2 = nr_ch3
        if self.hydrate_c1_carbon:
            reduce_ch2 += 1
        for nc in range(*self._ch2_calc_bounds):
            alkane = component(GC={"CH3": nr_ch3, "CH2": nc - reduce_ch2})
            mix = mixture(water, alkane)
            mix.saftgammamie()
            eos = saftgammamie(mix)
            mu = -1.0 * eos.get_lngamma(1.0 - 1e-10, T, self.pressure)[-1]
            self._mu_alkane.append(mu)
            if nc == self.tail_length:
                self.mu_alkane = mu

        m, _ = np.polyfit(range(*self._ch2_calc_bounds), self._mu_alkane, 1)
        self.mu_ch2 = m

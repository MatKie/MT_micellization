try:
    from SGTPy import saftvrmie, mixture, component
except:
    from sgtpy import saftvrmie, mixture, component
import numpy as np


class TransferSaft(object):
    def __init__(self):
        self.__water = component(
            "water",
            ms=1.7311,
            sigma=2.4539,
            eps=110.85,
            lambda_r=8.308,
            lambda_a=6.0,
            eAB=1991.07,
            rcAB=0.5624,
            rdAB=0.4,
            sites=[0, 2, 2],
            cii=1.5371939421515458e-20,
        )
        self.__PentaneComponent = component(
            "pentane", ms=2, sigma=4.248, eps=317.5, lambda_r=16.06, lambda_a=6.0
        )
        self.__HexaneComponent = component(
            "hexane", ms=2, sigma=4.508, eps=376.35, lambda_r=19.57, lambda_a=6.0
        )
        self.__HeptaneComponent = component(
            "heptane", ms=2, sigma=4.766, eps=436.13, lambda_r=23.81, lambda_a=6.0
        )
        self.__OctaneComponent = component(
            "octane", ms=3.0, sigma=4.227, eps=333.7, lambda_r=16.14, lambda_a=6.0
        )
        self.__NonaneComponent = component(
            "nonane", ms=3, sigma=4.406, eps=374.21, lambda_r=18.31, lambda_a=6.0
        )
        self.__DecaneComponent = component(
            "dodecane", ms=3, sigma=4.584, eps=415.19, lambda_r=20.92, lambda_a=6.0
        )
        self.__UndecaneComponent = component(
            "undecane", ms=4, sigma=4.216, eps=348.9, lambda_r=16.84, lambda_a=6.0
        )
        self.__components = [
            self.__PentaneComponent,
            self.__HexaneComponent,
            self.__HeptaneComponent,
            self.__OctaneComponent,
            self.__NonaneComponent,
            self.__DecaneComponent,
            self.__UndecaneComponent,
        ]
        # Sorry this is ugly, they should match the above sequence
        self.__kij_coeffs = [
            [-3.43303335e-06, 2.65261499e-03, -4.42357526e-01],
            [-3.61129302e-06, 2.76461939e-03, -4.71873413e-01],
            [-3.76062271e-06, 2.85952765e-03, -5.06227204e-01],
            [-3.76764999e-06, 2.88068999e-03, -4.70588225e-01],
            [-3.93111896e-06, 2.99157273e-03, -4.94123059e-01],
            [-4.09541375e-06, 3.10514750e-03, -5.22152479e-01],
            [-4.10217229e-06, 3.13499002e-03, -5.03836509e-01],
        ]

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
        for kij_coeffs, alkane in zip(self.__kij_coeffs, self.__components):
            p = np.poly1d(kij_coeffs)
            kij = p(T)
            kij_arr = np.array([[0.0, kij], [kij, 0.0]])
            mix = mixture(self.__water, alkane)
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

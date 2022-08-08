from ._saftvr_mie import SaftVR
from sgtpy import mixture, saftvrmie
from sgtpy.equilibrium import lle, lle_init
from sgtpy.sgt import sgt_mix
from numpy import poly1d, array
from numpy.linalg import LinAlgError


class SigmaSGT(SaftVR):
    interfacial_tension_vr = {}

    def __init__(self, tail_carbons):
        super().__init__()
        self.p = 101325
        self.tail_carbons = tail_carbons
        if (index := self.tail_carbons - 5) < len(self._SaftVR__components):
            pass
        else:
            index = len(self._SaftVR__components) - 1
        self.component = self._SaftVR__components[index]
        kij_polynomial = self._SaftVR__kij_coeffs[index]
        self.kij_polynomial = poly1d(kij_polynomial)
        bij_polynomial = self._SaftVR__bij_coeffs[index]
        self.bij_polynomial = poly1d(bij_polynomial)
        self.eos = None
        self.rho_l1, self.rho_l2 = None, None
        self._T = None
        self.itmax = 25

        # sgt_mix parameters
        self.z0 = 20
        self.rho0 = "hyperbolic"  # options are 'hyperbolic' and a previous result
        self.max_dz0 = 25

    @property
    def temperature(self):
        return self._T

    @temperature.setter
    def temperature(self, T):
        if T == None or T == self._T:
            pass
        elif T != self._T:
            # setup mixture+eos with lle calc and set beta matrix
            self._T = T
            self.eos = self._set_eos()
            self.rho_l1, self.rho_l2 = self._get_lle(self.eos)

    def get_ift(self, **kwargs):
        ift = SigmaSGT.interfacial_tension_vr.get(self.tail_carbons, {}).get(
            self.temperature
        )
        if ift == None:
            ift = self._get_ift(**kwargs)
            SigmaSGT.interfacial_tension_vr.update(
                {self.tail_carbons: {self.temperature: ift}}
            )

        return ift

    def _get_ift(self, **kwargs):
        rho0 = kwargs.get("rho0", self.rho0)
        z0 = kwargs.get("z0", self.z0)
        itmax = kwargs.get("itmax", self.itmax)
        sol = sgt_mix(
            self.rho_l1,
            self.rho_l2,
            self._T,
            self.p,
            self.eos,
            z0=z0,
            rho0=rho0,
            full_output=True,
            itmax=itmax,
        )
        dz0 = 0
        while not sol.success and dz0 < self.max_dz0:
            dz0 += 5
            try:
                sol = sgt_mix(
                    self.rho_l1,
                    self.rho_l2,
                    self._T,
                    self.p,
                    z0=self.z0 + dz0,
                    rho0=sol,
                    full_output=True,
                    itmax=itmax,
                )
            except LinAlgError:
                pass

        if not sol.success:
            raise RuntimeError("Could not get interfacial tension from SGT+SAFT-VR Mie")

        return sol.tension

    def _set_eos(self):
        mix = mixture(self._SaftVR__water, self.component)
        kij = self.kij_polynomial(self._T)
        Kij = array([[0.0, kij], [kij, 0.0]])
        mix.kij_saft(Kij)
        eos = saftvrmie(mix)

        bij = self.bij_polynomial(self._T)
        Bij = array([[0.0, bij], [bij, 0.0]])
        eos.beta_sgt(Bij)

        return eos

    def _get_lle(self, eos):
        x_wr, x_ar = array([1e-8, 1.0 - 1e-8]), array([1.0 - 1e-4, 1e-4])
        x_wr, x_ar = lle_init((x_wr + x_ar) / 2.0, self._T, self.p, eos)
        sol = lle(
            x_wr, x_ar, (x_wr + x_ar) / 2.0, self._T, self.p, eos, full_output=True
        )
        x_wr, x_ar = sol.X
        vl_wr, vl_ar = sol.v

        return x_wr / vl_wr, x_ar / vl_ar


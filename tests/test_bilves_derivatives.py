from MTM.micelles.bilayer_vesicle import BilayerVesicle
from MTM.micelles._bilayer_vesicle_derivative import BilayerVesicleDerivative
from numpy.core.numeric import roll
import pytest
import sys
import numpy as np
import os
from mkutils import create_fig, save_to_file
from MTM import literature
import MTM.micelles as micelle

sys.path.append("../")
this_path = os.path.dirname(__file__)


def absolute_difference(mic, mic_prop, mic_fun, h=1e-8):
    org_value = mic_fun(mic)
    mic_prop(mic, h)
    new_value = mic_fun(mic)

    deriv = (new_value - org_value) / h
    return deriv


def update_r(mic, h):
    mic._r_out += h


def update_t(mic, h):
    mic._t_out += h


class TestRodlikeMicelleDerivativeAbsoluteDifferences:
    def setup(self):
        self.mic = micelle.BilayerVesicle(80, 298.15, 10, throw_errors=False)
        self.mic._r_out = 1.8099053835829944
        self.mic._t_out = 0.8414754023196881
        self.d_mic = BilayerVesicleDerivative(self.mic)

    def test_deriv_r_in_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_r_in_wrt_r_out

        num_deriv = absolute_difference(self.mic, update_r, lambda x: x.radius_inner)

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_g_out_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_surfactants_number_out_wrt_r_out

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x.surfactants_number_outer
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_g_out_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_surfactants_number_out_wrt_t_out

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x.surfactants_number_outer
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_g_in_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_surfactants_number_in_wrt_r_out

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x.surfactants_number_inner
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_g_in_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_surfactants_number_in_wrt_t_out

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x.surfactants_number_inner
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_t_in_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_thickness_in_wrt_r_out

        num_deriv = absolute_difference(self.mic, update_r, lambda x: x.thickness_inner)

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_t_in_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_thickness_in_wrt_t_out

        num_deriv = absolute_difference(self.mic, update_t, lambda x: x.thickness_inner)

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_area_out_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_out_wrt_r_out

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x.area_per_surfactant_outer
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_area_out_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_out_wrt_t_out

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x.area_per_surfactant_outer
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_area_in_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_in_wrt_r_out

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x.area_per_surfactant_inner
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=2e-6)

    def test_area_in_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_in_wrt_t_out

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x.area_per_surfactant_inner
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_deformation_out_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_deformation_nagarajan_out_wrt_r_out()

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x._deformation_nagarajan_out()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_deformation_out_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_deformation_nagarajan_out_wrt_t_out()

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x._deformation_nagarajan_out()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_deformation_out_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_deformation_nagarajan_in_wrt_r_out()

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x._deformation_nagarajan_in()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=5e-6)

    def test_deformation_out_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_deformation_nagarajan_in_wrt_t_out()

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x._deformation_nagarajan_in()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_deformation_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_deformation_nagarajan_wrt_r_out()

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x._deformation_nagarajan()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_deformation_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_deformation_nagarajan_wrt_t_out()

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x._deformation_nagarajan()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_area_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_per_surfactant_wrt_r_out

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x.area_per_surfactant
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_area_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_per_surfactant_wrt_t_out

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x.area_per_surfactant
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_interface_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_interface_free_energy_wrt_r_out()

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x.get_interface_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=2e-6)

    def test_interface_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_interface_free_energy_wrt_t_out()

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x.get_interface_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    @pytest.mark.xfail
    def test_steric_wrt_r_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_steric_vdw_wrt_r_out()

        num_deriv = absolute_difference(
            self.mic, update_r, lambda x: x.get_steric_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-5)

    def test_steric_wrt_t_out(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_steric_vdw_wrt_t_out()

        num_deriv = absolute_difference(
            self.mic, update_t, lambda x: x.get_steric_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_jacobian(self):
        self.setup()
        self.mic._r_out += 0.05
        self.mic._t_out -= 0.01
        ana_rad, ana_thi = self.d_mic.jacobian()

        num_rad = absolute_difference(
            self.mic, update_r, lambda x: x.get_delta_chempot()
        )
        self.mic._r_out -= 1e-8

        num_thi = absolute_difference(
            self.mic, update_t, lambda x: x.get_delta_chempot()
        )

        assert ana_thi == pytest.approx(num_thi, abs=1e-6)
        assert ana_rad == pytest.approx(num_rad, abs=1e-6)

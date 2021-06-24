from MTM.micelles.rodlike_micelle import RodlikeMicelle
from MTM.micelles._rodlike_derivative import RodlikeMicelleDerivative
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


def update_sph(mic, h):
    mic._r_sph += h


def update_cyl(mic, h):
    mic._r_cyl += h


class TestRodlikeMicelleDerivativeAbsoluteDifferences:
    def setup(self):
        self.mic = micelle.RodlikeMicelle.optimised_radii(
            80, 298.15, 10, throw_errors=False
        )
        self.mic._r_sph += 0.1
        self.mic._r_cyl -= 0.1
        self.d_mic = RodlikeMicelleDerivative(
            self.mic, self.mic.radius_sphere, self.mic.radius_cylinder
        )

    def test_cap_height_deriv_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_cap_height_wrt_r_sph

        num_deriv = absolute_difference(self.mic, update_sph, lambda x: x.cap_height)

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_cap_height_deriv_wrt_r_cyl(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_cap_height_wrt_r_cyl

        num_deriv = absolute_difference(self.mic, update_cyl, lambda x: x.cap_height)

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_surfactants_cap_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_surfactants_number_cap_wrt_r_sph

        num_deriv = absolute_difference(
            self.mic, update_sph, lambda x: x.surfactants_number_cap
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_surfactants_cap_wrt_r_cyl(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_surfactants_number_cap_wrt_r_cyl

        num_deriv = absolute_difference(
            self.mic, update_cyl, lambda x: x.surfactants_number_cap
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_surfactants_cyl_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_surfactants_number_cyl_wrt_r_sph

        num_deriv = absolute_difference(
            self.mic, update_sph, lambda x: x.surfactants_number_cyl
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_surfactants_cyl_wrt_r_cyl(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_surfactants_number_cyl_wrt_r_cyl

        num_deriv = absolute_difference(
            self.mic, update_cyl, lambda x: x.surfactants_number_cyl
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-6)

    def test_length_cyl_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_cylinder_length_wrt_r_sph

        num_deriv = absolute_difference(
            self.mic, update_sph, lambda x: x.cylinder_length
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_length_cyl_wrt_r_cyl(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_cylinder_length_wrt_r_cyl

        num_deriv = absolute_difference(
            self.mic, update_cyl, lambda x: x.cylinder_length
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_area_per_surfactant_cyl_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_per_surfactant_cyl_wrt_r_sph

        num_deriv = absolute_difference(
            self.mic, update_sph, lambda x: x.area_per_surfactant_cyl
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_area_per_surfactant_cyl_wrt_r_cyl(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_per_surfactant_cyl_wrt_r_cyl

        num_deriv = absolute_difference(
            self.mic, update_cyl, lambda x: x.area_per_surfactant_cyl
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_area_per_surfactant_cap_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_per_surfactant_cap_wrt_r_sph

        num_deriv = absolute_difference(
            self.mic, update_sph, lambda x: x.area_per_surfactant_cap
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_area_per_surfactant_cap_wrt_r_cyl(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_area_per_surfactant_cap_wrt_r_cyl

        num_deriv = absolute_difference(
            self.mic, update_cyl, lambda x: x.area_per_surfactant_cap
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_interface_deriv_wrt_r_cyl(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_interface_free_energy_wrt_r_cyl()

        num_deriv = absolute_difference(
            self.mic, update_cyl, lambda x: x.get_interface_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_interface_deriv_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_interface_free_energy_wrt_r_sph()

        num_deriv = absolute_difference(
            self.mic, update_sph, lambda x: x.get_interface_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_deformation_nagarajan_deriv_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_deformation_nagarajan_wrt_r_sph()

        num_deriv = absolute_difference(
            self.mic, update_sph, lambda x: x.get_deformation_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_deformation_nagarajan_deriv_wrt_r_cyl(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_deformation_nagarajan_wrt_r_cyl()

        num_deriv = absolute_difference(
            self.mic, update_cyl, lambda x: x.get_deformation_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_steric_vdw_cap_deriv_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_steric_vdw_cap_wrt_r_sph()

        num_deriv = absolute_difference(self.mic, update_sph, lambda x: x._steric_sph())

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_steric_vdw_cap_deriv_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_steric_vdw_cap_wrt_r_cyl()

        num_deriv = absolute_difference(self.mic, update_cyl, lambda x: x._steric_sph())

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_steric_vdw_cyl_deriv_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_steric_vdw_cyl_wrt_r_sph()

        num_deriv = absolute_difference(self.mic, update_sph, lambda x: x._steric_cyl())

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_steric_vdw_cyl_deriv_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_steric_vdw_cyl_wrt_r_sph()

        num_deriv = absolute_difference(self.mic, update_sph, lambda x: x._steric_cyl())

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_steric_vdw_deriv_wrt_r_sph(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_steric_vdw_wrt_r_sph()

        num_deriv = absolute_difference(
            self.mic, update_sph, lambda x: x.get_steric_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_steric_vdw_deriv_wrt_r_cyl(self):
        self.setup()
        ana_deriv = self.d_mic.deriv_steric_vdw_wrt_r_cyl()

        num_deriv = absolute_difference(
            self.mic, update_cyl, lambda x: x.get_steric_free_energy()
        )

        assert ana_deriv == pytest.approx(num_deriv, abs=1e-7)

    def test_jacobian(self):
        self.setup()
        ana_sph, ana_cyl = self.d_mic.jacobian()

        num_sph = absolute_difference(
            self.mic, update_sph, lambda x: x.get_delta_chempot()
        )
        self.mic._r_sph -= 1e-8

        num_cyl = absolute_difference(
            self.mic, update_cyl, lambda x: x.get_delta_chempot()
        )

        assert ana_sph == pytest.approx(num_sph, abs=1e-6)
        assert ana_cyl == pytest.approx(num_cyl, abs=1e-6)

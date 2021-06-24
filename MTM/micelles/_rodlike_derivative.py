import numpy as np


class RodlikeMicelleDerivative(object):
    def __init__(self, base_micelle, r_sphere, r_cylinder):
        self.base_micelle = base_micelle
        self._r_sph = r_sphere
        self._r_cyl = r_cylinder

    @property
    def deriv_cap_height_wrt_r_sph(self):
        """
        Derivative of H wrt. radius of sphere
        """
        rc = self.base_micelle.radius_cylinder
        rs = self.base_micelle.radius_sphere
        rc_sq = rc * rc
        rs_sq = rs * rs

        ratio_sq = rc_sq / rs_sq

        term_1 = 1.0 - np.sqrt(1.0 - ratio_sq)
        term_2 = -1.0 * ratio_sq / np.sqrt(1.0 - ratio_sq)

        return term_1 + term_2

    @property
    def deriv_cap_height_wrt_r_cyl(self):
        """
        Derivative of H wrt. radius of cylinder
        """
        rc = self.base_micelle.radius_cylinder
        rs = self.base_micelle.radius_sphere
        rc_sq = rc * rc
        rs_sq = rs * rs

        ratio_sq = rc_sq / rs_sq

        term_1 = rc / (rs * np.sqrt(1.0 - ratio_sq))

        return term_1

    @property
    def deriv_surfactants_number_cap_wrt_r_sph(self):
        """
        Derivative of surfactants in cap wrt. to r_sphere
        """
        H = self.base_micelle.cap_height
        dH_drs = self.deriv_cap_height_wrt_r_sph
        rs = self.base_micelle.radius_sphere
        volume = self.base_micelle.volume

        prefac = 2.0 * np.pi / (3.0 * volume)
        term_1 = 12.0 * rs * rs
        term_2 = -2.0 * H * dH_drs * (3.0 * rs * rs * rs - H)
        term_3 = -1.0 * H * H * (3.0 - dH_drs)

        return prefac * (term_1 + term_2 + term_3)

    @property
    def deriv_surfactants_number_cap_wrt_r_cyl(self):
        """
        Derivative of surfactants in cap wrt. to r_cylinder
        """
        H = self.base_micelle.cap_height
        dH_drc = self.deriv_cap_height_wrt_r_cyl
        rs = self.base_micelle.radius_sphere
        volume = self.base_micelle.volume

        prefac = 2.0 * np.pi / (3.0 * volume)
        term_1 = 0.0
        term_2 = -2.0 * H * dH_drc * (3.0 * rs * rs * rs - H)
        term_3 = -1.0 * H * H*(-1.0 * dH_drc)

        return prefac * (term_1 + term_2 + term_3)

    @property
    def deriv_surfactants_number_cyl_wrt_r_sph(self):
        """
        Derivative of surfactants in cap wrt. to r_cylinder
        """
        return -1.0 * self.deriv_surfactants_number_cap_wrt_r_sph

    @property
    def deriv_surfactants_number_cyl_wrt_r_cyl(self):
        """
        Derivative of surfactants in cylinder wrt. r_sphere
        """
        return -1.0 * self.deriv_surfactants_number_cap_wrt_r_cyl

    @property
    def deriv_cylinder_length_wrt_r_sph(self):
        """
        Derivative of cylinder length wrt. r_sphere
        """
        volume = self.base_micelle.volume
        rc = self.base_micelle.radius_cylinder
        dgcyl_drs = self.deriv_surfactants_number_cyl_wrt_r_sph
        prefac = volume / np.pi

        term_1 = rc * rc * dgcyl_drs

        return prefac * term_1

    @property
    def deriv_cylinder_length_wrt_r_cyl(self):
        """
        Derivative of cylinder length wrt. r_sphere
        """
        volume = self.base_micelle.volume
        rc = self.base_micelle.radius_cylinder
        dgcyl_drc = self.deriv_surfactants_number_cyl_wrt_r_cyl
        gcyl = self.base_micelle.surfactants_number_cyl
        prefac = volume / np.pi

        term_1 = rc * rc * dgcyl_drc
        term_2 = gcyl * 2.0 * rc

        return prefac * (term_1 + term_2)

    @property
    def deriv_area_per_surfactant_cyl_wrt_r_cyl(self):
        """
        Derivative of cylinder area per surfactant wrt r_cylinder
        """
        rc = self.base_micelle.radius_cylinder
        lc = self.base_micelle.cylinder_length
        gcyl = self.base_micelle.surfactants_number_cyl
        dlc_drc = self.deriv_cylinder_length_wrt_r_cyl
        dgcyl_drc = self.deriv_surfactants_number_cyl_wrt_r_cyl

        term_1 = lc * gcyl
        term_2 = rc * dlc_drc * gcyl
        term_3 = rc * lc * dgcyl_drc

        return np.pi * (term_1 + term_2 + term_3)

    @property
    def deriv_area_per_surfactant_cyl_wrt_r_sph(self):
        """
        Derivative of cylinder area per surfactant wrt r_sphere
        """
        rc = self.base_micelle.radius_cylinder
        lc = self.base_micelle.cylinder_length
        gcyl = self.base_micelle.surfactants_number_cyl
        dlc_drs = self.deriv_cylinder_length_wrt_r_sph
        dgcyl_drs = self.deriv_surfactants_number_cyl_wrt_r_sph

        term_2 = rc * dlc_drs * gcyl
        term_3 = rc * lc * dgcyl_drs

        return np.pi * (term_2 + term_3)

    @property
    def deriv_area_per_surfactant_cap_wrt_r_sph(self):
        """
        Derivative of sphere area per surfactant wrt r_sphere
        """
        H = self.base_micelle.cap_height
        rs = self.base_micelle.radius_sphere
        gcap = self.base_micelle.surfactants_number_cap
        dH_drs = self.deriv_cap_height_wrt_r_sph
        dgcap_drs = self.deriv_surfactants_number_cap_wrt_r_sph

        aux = 2.0 * rs - H
        term_1 = gcap * aux
        term_2 = rs * dgcap_drs * aux
        term_3 = rs * gcap * (2.0 - dH_drs)

        return 4.0 * np.pi * (term_1 + term_2 + term_3)

    @property
    def deriv_area_per_surfactant_cap_wrt_r_cyl(self):
        """
        Derivative of sphere area per surfactant wrt r_sphere
        """
        H = self.base_micelle.cap_height
        rs = self.base_micelle.radius_sphere
        gcap = self.base_micelle.surfactants_number_cap
        dH_drc = self.deriv_cap_height_wrt_r_cyl
        dgcap_drc = self.deriv_surfactants_number_cap_wrt_r_cyl

        aux = 2.0 * rs - H
        term_2 = rs * dgcap_drc * aux
        term_3 = -1.0 * rs * gcap * dH_drc

        return 4.0 * np.pi * (term_2 + term_3)

    @property
    def deriv_area_per_surfactant_wrt_r_sph(self):
        a_cyl = self.base_micelle.area_per_surfactant_cyl
        da_cyl_drs = self.deriv_area_per_surfactant_cyl_wrt_r_sph
        a_cap = self.base_micelle.area_per_surfactant_cap
        da_cap_drs = self.deriv_area_per_surfactant_cap_wrt_r_sph
        g_cyl = self.base_micelle.surfactants_number_cyl
        dg_cyl_drs = self.deriv_surfactants_number_cyl_wrt_r_sph
        g_cap = self.base_micelle.surfactants_number_cap
        dg_cap_drs = self.deriv_surfactants_number_cap_wrt_r_sph

        area = (
            (a_cyl * dg_cyl_drs)
            + (da_cyl_drs * g_cyl)
            + (a_cap * dg_cap_drs)
            + (da_cap_drs * g_cap)
        )

        return area / (g_cap + g_cyl)

    @property
    def deriv_area_per_surfactant_wrt_r_cyl(self):
        a_cyl = self.base_micelle.area_per_surfactant_cyl
        da_cyl_drc = self.deriv_area_per_surfactant_cyl_wrt_r_cyl
        a_cap = self.base_micelle.area_per_surfactant_cap
        da_cap_drc = self.deriv_area_per_surfactant_cap_wrt_r_cyl
        g_cyl = self.base_micelle.surfactants_number_cyl
        dg_cyl_drc = self.deriv_surfactants_number_cyl_wrt_r_cyl
        g_cap = self.base_micelle.surfactants_number_cap
        dg_cap_drc = self.deriv_surfactants_number_cap_wrt_r_cyl

        area = (
            (a_cyl * dg_cyl_drc)
            + (da_cyl_drc * g_cyl)
            + (a_cap * dg_cap_drc)
            + (da_cap_drc * g_cap)
        )

        return area / (g_cap + g_cyl)

    def deriv_deformation_nagarajan_wrt_r_sph(self):
        surfactants_number = self.base_micelle.surfactants_number
        g_cap = self.base_micelle.surfactants_number_cap
        g_cyl = surfactants_number - g_cap
        dg_cyl_drs = self.deriv_surfactants_number_cyl_wrt_r_sph()
        dg_cap_drs = self.deriv_surfactants_number_cap_wrt_r_sph()
        deformation_cyl = self.base_micelle._deformation_nagarajan_cyl()
        deformation_sph = self.base_micelle._deformation_nagarajan_sph()
        d_deformation_cyl = 0.0
        d_deformation_sph = self.deriv_deformation_nagarajan_cap_wrt_r_sph()

        term_cyl = g_cyl * d_deformation_cyl + dg_cyl_drs * deformation_cyl
        term_sph = g_cap * d_deformation_sph + dg_cap_drs * deformation_sph

        return (term_cyl + term_sph) / surfactants_number

    def deriv_deformation_nagarajan_wrt_r_cyl(self):
        surfactants_number = self.base_micelle.surfactants_number
        g_cap = self.base_micelle.surfactants_number_cap
        g_cyl = surfactants_number - g_cap
        dg_cyl_drc = self.deriv_surfactants_number_cyl_wrt_r_cyl()
        dg_cap_drc = self.deriv_surfactants_number_cap_wrt_r_cyl()
        deformation_cyl = self.base_micelle._deformation_nagarajan_cyl()
        deformation_sph = self.base_micelle._deformation_nagarajan_sph()
        d_deformation_cyl = self.deriv_deformation_nagarajan_cyl_wrt_r_cyl()
        d_deformation_sph = self.deriv_deformation_nagarajan_cap_wrt_r_cyl()

        term_cyl = g_cyl * d_deformation_cyl + dg_cyl_drc * deformation_cyl
        term_sph = g_cap * d_deformation_sph + dg_cap_drc * deformation_sph

        return (term_cyl + term_sph) / surfactants_number

    def deriv_deformation_nagarajan_cap_wrt_r_sph(self):
        volume = self.base_micelle.volume
        segment_length = self.base_micelle.segment_length
        length = self.base_micelle.lengths
        rs = self.base_micelle.radius_sphere
        area = self.base_micelle.area_per_surfactant_cap
        da_drs = self.deriv_area_per_surfactant_cap_wrt_r_sph

        prefac = 9.0 * volume * np.pi * np.pi / (80.0 * length * segment_length)

        term_1 = 1.0 / area
        term_2 = -1.0 * rs * da_drs / (area * area)

        return prefac * (term_1 + term_2)

    def deriv_deformation_nagarajan_cap_wrt_r_cyl(self):
        volume = self.base_micelle.volume
        segment_length = self.base_micelle.segment_length
        length = self.base_micelle.length
        rs = self.base_micelle.radius_sphere
        area = self.base_micelle.area_per_surfactant_cap
        da_drc = self.deriv_area_per_surfactant_cap_wrt_r_cyl

        prefac = 9.0 * volume * np.pi * np.pi / (80.0 * length * segment_length)

        term_2 = -1.0 * rs * da_drc / (area * area)

        return prefac * term_2

    def deriv_deformation_nagarajan_cyl_wrt_r_cyl(self):
        segment_length = self.base_micelle.segment_length
        length = self.base_micelle.length
        rc = self.base_micelle.radius_cylinder

        prefac = 5.0 * np.pi * np.pi / (80.0 * length * segment_length)

        term_1 = 2 * rc

        return prefac * term_1

    def deriv_steric_vdw_cap_wrt_r_sph(self):
        headgroup_area = self.base_micelle.headgroup_area
        area_cap = self.base_micelle.area_per_surfactant_cap
        d_area_drs = self.deriv_area_per_surfactant_cap_wrt_r_sph()

        factor = headgroup_area / (headgroup_area - area_cap)
        deriv = factor * d_area_drs

        return deriv

    def deriv_steric_vdw_cap_wrt_r_cyl(self):
        headgroup_area = self.base_micelle.headgroup_area
        area_cap = self.base_micelle.area_per_surfactant_cap
        d_area_drc = self.deriv_area_per_surfactant_cap_wrt_r_cyl()

        factor = headgroup_area / (headgroup_area - area_cap)
        deriv = factor * d_area_drc

        return deriv

    def deriv_steric_vdw_cyl_wrt_r_sph(self):
        headgroup_area = self.base_micelle.headgroup_area
        area_cyl = self.base_micelle.area_per_surfactant_cyl
        d_area_drs = self.deriv_area_per_surfactant_cyl_wrt_r_sph()

        factor = headgroup_area / (headgroup_area - area_cyl)
        deriv = factor * d_area_drs

        return deriv

    def deriv_steric_vdw_cyl_wrt_r_cyl(self):
        headgroup_area = self.base_micelle.headgroup_area
        area_cyl = self.base_micelle.area_per_surfactant_cyl
        d_area_drc = self.deriv_area_per_surfactant_cyl_wrt_r_cyl()

        factor = headgroup_area / (headgroup_area - area_cyl)
        deriv = factor * d_area_drc

        return deriv

    def deriv_steric_vdw_wrt_r_sph(self):
        g_cap = self.base_micelle.surfactants_number_cap
        g_cyl = self.base_micelle.surfactants_number_cyl
        d_g_cyl_drs = self.deriv_surfactants_number_cyl_wrt_r_sph()
        d_g_cap_drs = self.deriv_surfactants_number_cap_wrt_r_sph()

        steric_cap = self.base_micelle._steric_cap()
        d_steric_cap = self.deriv_steric_vdw_cap_wrt_r_sph()
        steric_cyl = self.base_micelle._steric_cyl()
        d_steric_cyl = self.deriv_steric_vdw_cyl_wrt_r_sph()

        cap = steric_cap * d_g_cap_drs + d_steric_cap * g_cap
        cyl = steric_cyl * d_g_cyl_drs + d_steric_cyl * g_cyl

        deform = cap + cyl

        return deform

    def deriv_steric_vdw_wrt_r_cyl(self):
        g_cap = self.base_micelle.surfactants_number_cap
        g_cyl = self.base_micelle.surfactants_number_cyl
        d_g_cyl_drc = self.deriv_surfactants_number_cyl_wrt_r_cyl()
        d_g_cap_drc = self.deriv_surfactants_number_cap_wrt_r_cyl()

        steric_cap = self.base_micelle._steric_cap()
        d_steric_cap = self.deriv_steric_vdw_cap_wrt_r_cyl()
        steric_cyl = self.base_micelle._steric_cyl()
        d_steric_cyl = self.deriv_steric_vdw_cyl_wrt_r_cyl()

        cap = steric_cap * d_g_cap_drc + d_steric_cap * g_cap
        cyl = steric_cyl * d_g_cyl_drc + d_steric_cyl * g_cyl

        deform = cap + cyl

        return deform

    def deriv_interface_free_energy_wrt_r_sph(self):
        sigma = self.base_micelle._sigma_agg()
        d_area_drs = self.deriv_area_per_surfactant_wrt_r_sph

        return sigma * d_area_drs

    def deriv_interface_free_energy_wrt_r_cyl(self):
        sigma = self.base_micelle._sigma_agg()
        d_area_drc = self.deriv_area_per_surfactant_wrt_r_cyl

        return sigma * d_area_drc

    def jacobian(self):
        # rsph
        sph_steric = self.deriv_steric_vdw_wrt_r_sph()
        sph_deform = self.deriv_deformation_nagarajan_wrt_r_sph()
        sph_interface = self.deriv_interface_free_energy_wrt_r_sph()
        rsph = sph_steric + sph_interface + sph_deform

        # rcyl
        cyl_steric = self.deriv_steric_vdw_wrt_r_sph()
        cyl_deform = self.deriv_deformation_nagarajan_wrt_r_sph()
        cyl_interface = self.deriv_interface_free_energy_wrt_r_sph()

        rcyl = cyl_steric + cyl_interface + cyl_deform

        return np.asarray([rsph, rcyl])

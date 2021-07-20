import numpy as np
from numpy.lib.histograms import _ravel_and_check_weights


class BilayerVesicleDerivative(object):
    def __init__(self, base_micelle, r_out=None, t_out=None):
        self.base_micelle = base_micelle
        if r_out is not None:
            self.base_micelle._r_out = r_out
        if t_out is not None:
            self.base_micelle._t_out = t_out

        self.derivative_wrt_r_out = None
        self.derivative_wrt_t_out = None

    @property
    def deriv_r_in_wrt_r_out(self):
        r_out = self.base_micelle.radius_outer
        volume = self.base_micelle.volume
        g_overall = self.base_micelle.surfactants_number

        r_3 = r_out * r_out * r_out
        aux = 3.0 * g_overall * volume / (4.0 * np.pi)
        p_1 = np.power((r_3 - aux), -2.0 / 3.0)
        p_2 = r_out * r_out  # times three handled in p_1

        product = p_1 * p_2
        return product

    @property
    def deriv_surfactants_number_out_wrt_r_out(self):
        r_out = self.base_micelle.radius_outer
        t_out = self.base_micelle.thickness_outer
        volume = self.base_micelle.volume

        factor = 4.0 * np.pi / (3.0 * volume)
        s_1 = 3.0 * r_out * r_out
        s_2 = -3.0 * (r_out - t_out) * (r_out - t_out)

        deriv = factor * (s_1 + s_2)
        return deriv

    @property
    def deriv_surfactants_number_out_wrt_t_out(self):
        r_out = self.base_micelle.radius_outer
        t_out = self.base_micelle.thickness_outer
        volume = self.base_micelle.volume

        factor = 4.0 * np.pi / (3.0 * volume)
        s_1 = 3.0 * (r_out - t_out) * (r_out - t_out)

        deriv = factor * s_1
        return deriv

    @property
    def deriv_surfactants_number_in_wrt_r_out(self):
        return -1.0 * self.deriv_surfactants_number_out_wrt_r_out

    @property
    def deriv_surfactants_number_in_wrt_t_out(self):
        return -1.0 * self.deriv_surfactants_number_out_wrt_t_out

    @property
    def deriv_thickness_in_wrt_r_out(self):
        volume = self.base_micelle.volume
        r_in = self.base_micelle.radius_inner
        g_in = self.base_micelle.surfactants_number_inner
        d_g_in_wrt_r_out = self.deriv_surfactants_number_in_wrt_r_out
        d_r_in_wrt_r_out = self.deriv_r_in_wrt_r_out

        factor = 3.0 * volume / (4.0 * np.pi)
        aux = factor * g_in + r_in * r_in * r_in
        s_1 = (
            1.0
            / 3.0
            * np.power(aux, -2.0 / 3.0)
            * (factor * d_g_in_wrt_r_out + 3.0 * r_in * r_in * d_r_in_wrt_r_out)
        )
        s_2 = -1.0 * d_r_in_wrt_r_out

        deriv = s_1 + s_2
        return deriv

    @property
    def deriv_thickness_in_wrt_t_out(self):
        volume = self.base_micelle.volume
        r_in = self.base_micelle.radius_inner
        g_in = self.base_micelle.surfactants_number_inner
        d_g_in_wrt_t_out = self.deriv_surfactants_number_in_wrt_t_out

        factor = 3.0 * volume / (4.0 * np.pi)
        aux = factor * g_in + r_in * r_in * r_in

        deriv = 1.0 / 3.0 * np.power(aux, -2.0 / 3.0) * factor * d_g_in_wrt_t_out

        return deriv

    @property
    def deriv_area_out_wrt_r_out(self):
        # we follow a_o = A_o/g_o here
        r_out = self.base_micelle.radius_outer
        g_out = self.base_micelle.surfactants_number_outer
        d_g_out_r_out = self.deriv_surfactants_number_out_wrt_r_out

        s_1 = 8.0 * np.pi * r_out * g_out
        s_2 = -d_g_out_r_out * 4.0 * np.pi * r_out * r_out
        nom = s_1 + s_2
        denom = g_out * g_out

        deriv = nom / denom
        return deriv

    @property
    def deriv_area_out_wrt_t_out(self):
        # we follow a_o = A_o/g_o here
        r_out = self.base_micelle.radius_outer
        g_out = self.base_micelle.surfactants_number_outer
        d_g_out_t_out = self.deriv_surfactants_number_out_wrt_t_out

        s_2 = -d_g_out_t_out * 4.0 * np.pi * r_out * r_out
        nom = s_2
        denom = g_out * g_out

        deriv = nom / denom
        return deriv

    @property
    def deriv_area_in_wrt_r_out(self):
        # we follow a_i = A_i/g_i here
        r_in = self.base_micelle.radius_inner
        d_r_in_r_out = self.deriv_r_in_wrt_r_out
        g_in = self.base_micelle.surfactants_number_inner
        d_g_in_r_out = self.deriv_surfactants_number_in_wrt_r_out

        s_1 = 8.0 * d_r_in_r_out * r_in * np.pi * g_in
        s_2 = -d_g_in_r_out * 4.0 * np.pi * r_in * r_in
        nom = s_1 + s_2
        denom = g_in * g_in

        deriv = nom / denom

        return deriv

    @property
    def deriv_area_in_wrt_t_out(self):
        # we follow a_i = A_i/g_i here
        r_in = self.base_micelle.radius_inner
        g_in = self.base_micelle.surfactants_number_inner
        d_g_in_t_out = self.deriv_surfactants_number_in_wrt_t_out

        s_2 = -d_g_in_t_out * 4.0 * np.pi * r_in * r_in
        nom = s_2
        denom = g_in * g_in

        deriv = nom / denom

        return deriv

    @property
    def deriv_area_per_surfactant_wrt_r_out(self):
        a_in = self.base_micelle.area_per_surfactant_inner
        a_out = self.base_micelle.area_per_surfactant_outer
        g_in = self.base_micelle.surfactants_number_inner
        g_out = self.base_micelle.surfactants_number_outer

        d_a_in_r_out = self.deriv_area_in_wrt_r_out
        d_a_out_r_out = self.deriv_area_out_wrt_r_out
        d_g_in_r_out = self.deriv_surfactants_number_in_wrt_r_out
        d_g_out_r_out = self.deriv_surfactants_number_out_wrt_r_out

        s_1 = a_in * d_g_in_r_out + g_in * d_a_in_r_out
        s_2 = a_out * d_g_out_r_out + g_out * d_a_out_r_out

        deriv = (s_1 + s_2) / (g_in + g_out)

        return deriv

    @property
    def deriv_area_per_surfactant_wrt_t_out(self):
        a_in = self.base_micelle.area_per_surfactant_inner
        a_out = self.base_micelle.area_per_surfactant_outer
        g_in = self.base_micelle.surfactants_number_inner
        g_out = self.base_micelle.surfactants_number_outer

        d_a_in_t_out = self.deriv_area_in_wrt_t_out
        d_a_out_t_out = self.deriv_area_out_wrt_t_out
        d_g_in_t_out = self.deriv_surfactants_number_in_wrt_t_out
        d_g_out_t_out = self.deriv_surfactants_number_out_wrt_t_out

        s_1 = a_in * d_g_in_t_out + g_in * d_a_in_t_out
        s_2 = a_out * d_g_out_t_out + g_out * d_a_out_t_out

        deriv = (s_1 + s_2) / (g_in + g_out)

        return deriv

    def deriv_deformation_nagarajan_out_wrt_r_out(self):
        return 0

    def deriv_deformation_nagarajan_out_wrt_t_out(self):
        segment_length = self.base_micelle.segment_length
        length = self.base_micelle.length
        t_out = self.base_micelle.thickness_outer

        factor = 10.0 * np.pi * np.pi / (320.0 * segment_length * length)
        deriv = factor * 2.0 * t_out

        return deriv

    def deriv_deformation_nagarajan_in_wrt_r_out(self):
        segment_length = self.base_micelle.segment_length
        length = self.base_micelle.length
        t_in = self.base_micelle.thickness_inner
        d_t_in_r_out = self.deriv_thickness_in_wrt_r_out

        factor = 10.0 * np.pi * np.pi / (160.0 * segment_length * length)
        deriv = factor * 2.0 * t_in * d_t_in_r_out

        return deriv

    def deriv_deformation_nagarajan_in_wrt_t_out(self):
        segment_length = self.base_micelle.segment_length
        length = self.base_micelle.length
        t_in = self.base_micelle.thickness_inner
        d_t_in_t_out = self.deriv_thickness_in_wrt_t_out

        factor = 10.0 * np.pi * np.pi / (160.0 * segment_length * length)
        deriv = factor * 2.0 * t_in * d_t_in_t_out

        return deriv

    def deriv_deformation_nagarajan_wrt_r_out(self):
        g_in = self.base_micelle.surfactants_number_inner
        g_out = self.base_micelle.surfactants_number_outer
        d_g_in_r_out = self.deriv_surfactants_number_in_wrt_r_out
        d_g_out_r_out = self.deriv_surfactants_number_out_wrt_r_out
        deform_out = self.base_micelle._deformation_nagarajan_out()
        deform_in = self.base_micelle._deformation_nagarajan_in()
        d_def_out_r_out = self.deriv_deformation_nagarajan_out_wrt_r_out()
        d_def_in_r_out = self.deriv_deformation_nagarajan_in_wrt_r_out()

        outer = g_out * d_def_out_r_out + d_g_out_r_out * deform_out
        inner = g_in * d_def_in_r_out + d_g_in_r_out * deform_in

        deriv = (outer + inner) / (g_in + g_out)
        return deriv

    def deriv_deformation_nagarajan_wrt_t_out(self):
        g_in = self.base_micelle.surfactants_number_inner
        g_out = self.base_micelle.surfactants_number_outer
        d_g_in_t_out = self.deriv_surfactants_number_in_wrt_t_out
        d_g_out_t_out = self.deriv_surfactants_number_out_wrt_t_out
        deform_out = self.base_micelle._deformation_nagarajan_out()
        deform_in = self.base_micelle._deformation_nagarajan_in()
        d_def_out_t_out = self.deriv_deformation_nagarajan_out_wrt_t_out()
        d_def_in_t_out = self.deriv_deformation_nagarajan_in_wrt_t_out()

        outer = g_out * d_def_out_t_out + d_g_out_t_out * deform_out
        inner = g_in * d_def_in_t_out + d_g_in_t_out * deform_in

        deriv = (outer + inner) / (g_in + g_out)
        return deriv

    def deriv_interface_free_energy_wrt_r_out(self):
        sigma = self.base_micelle._sigma_agg()
        d_area_drs = self.deriv_area_per_surfactant_wrt_r_out

        return sigma * d_area_drs

    def deriv_interface_free_energy_wrt_t_out(self):
        sigma = self.base_micelle._sigma_agg()
        d_area_drs = self.deriv_area_per_surfactant_wrt_t_out

        return sigma * d_area_drs

    def deriv_steric_vdw_wrt_r_out(self):
        a_head = self.base_micelle.headgroup_area
        g_out = self.base_micelle.surfactants_number_outer
        g_in = self.base_micelle.surfactants_number_inner
        a_out = self.base_micelle.area_per_surfactant_outer
        a_in = self.base_micelle.area_per_surfactant_inner
        g_all = g_in + g_out

        d_g_out_r_out = self.deriv_surfactants_number_out_wrt_r_out
        d_g_in_r_out = self.deriv_surfactants_number_in_wrt_r_out
        d_a_out_r_out = self.deriv_area_out_wrt_r_out
        d_a_in_r_out = self.deriv_area_in_wrt_r_out

        s_11 = d_g_out_r_out * np.log(1.0 - a_head / a_out)
        s_12 = g_out / (1.0 - a_head / a_out) * d_a_out_r_out * a_head / (a_out * a_out)
        s_21 = d_g_in_r_out * np.log(1.0 - a_head / a_in)
        s_22 = g_in / (1.0 - a_head / a_in) * d_a_in_r_out * a_head / (a_in * a_in)

        deriv = -(s_11 + s_12 + s_21 + s_22) / g_all

        return deriv

    def deriv_steric_vdw_wrt_t_out(self):
        a_head = self.base_micelle.headgroup_area
        g_out = self.base_micelle.surfactants_number_outer
        g_in = self.base_micelle.surfactants_number_inner
        a_out = self.base_micelle.area_per_surfactant_outer
        a_in = self.base_micelle.area_per_surfactant_inner
        g_all = g_in + g_out

        d_g_out_t_out = self.deriv_surfactants_number_out_wrt_t_out
        d_g_in_t_out = self.deriv_surfactants_number_in_wrt_t_out
        d_a_out_t_out = self.deriv_area_out_wrt_t_out
        d_a_in_t_out = self.deriv_area_in_wrt_t_out

        s_11 = d_g_out_t_out * np.log(1.0 - a_head / a_out)
        s_12 = g_out / (1.0 - a_head / a_out) * d_a_out_t_out * a_head / (a_out * a_out)
        s_21 = d_g_in_t_out * np.log(1.0 - a_head / a_in)
        s_22 = g_in / (1.0 - a_head / a_in) * d_a_in_t_out * a_head / (a_in * a_in)

        deriv = -(s_11 + s_12 + s_21 + s_22) / g_all

        return deriv

    def jacobian(self, x=None):
        """
        Return the derivative of the chemical potential difference of
        a bilayer vesicle micelle with respect to the outer radius and outer
        thickness

        If the values are negative in either direction return a value
        giving an opposite search direction. I.e. returning a more negative
        or more positive value.

        Parameters
        ----------
        x : array like of length 2, optional
            Values at which to evaluate the derivatives.
            If None, take the current values of the micelle radii, by default None

        Returns
        -------
        numpy array, shape (2,)
            Derivatives wrt outer radius and outer thickness respectively.
        """
        if x is not None:
            self.base_micelle._r_out, self.base_micelle._t_out = x[0], x[1]
        else:
            x = np.zeros((2,))
            x[0], x[1] = (
                self.base_micelle.radius_outer,
                self.base_micelle.thickness_outer,
            )
        # Boundaries
        if (
            x[0] < 0
            or x[1] < 0
            or x[0] > 10
            or x[1] > 5
            or not self.base_micelle._check_geometry()
        ):
            # Return a higher value to give a negative gradient.
            sign_r_out = np.sign(self.derivative_wrt_r_out)
            sign_t_out = np.sign(self.derivative_wrt_t_out)
            return np.asarray(
                [
                    self.derivative_wrt_r_out + sign_r_out,
                    self.derivative_wrt_t_out + sign_t_out,
                ]
            )
        if x[0] < x[1]:
            # If thickness is higher than radius return a higher value to
            # get away from those solution.
            sign_t_out = np.sign(self.derivative_wrt_t_out)
            return np.asarray(
                [self.derivative_wrt_r_out, self.derivative_wrt_t_out + sign_t_out]
            )

        # r out
        rad_steric = self.deriv_steric_vdw_wrt_r_out()
        rad_deform = self.deriv_deformation_nagarajan_wrt_r_out()
        rad_interface = self.deriv_interface_free_energy_wrt_r_out()
        self.derivative_wrt_r_out = rad_steric + rad_interface + rad_deform

        # t out
        thi_steric = self.deriv_steric_vdw_wrt_t_out()
        thi_deform = self.deriv_deformation_nagarajan_wrt_t_out()
        thi_interface = self.deriv_interface_free_energy_wrt_t_out()

        self.derivative_wrt_t_out = thi_steric + thi_interface + thi_deform

        return np.asarray([self.derivative_wrt_r_out, self.derivative_wrt_t_out])

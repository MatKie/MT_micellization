import numpy as np
from numpy.lib.histograms import _ravel_and_check_weights


class BilayerVesicleDerivative(object):
    def __init__(self, base_micelle, r_out=None, t_out=None):
        self.base_micelle = base_micelle
        if r_out is not None:
            self.base_micelle._r_out = r_out
        if t_out is not None:
            self.base_micelle._t_out = t_out

        self.deriv_wrt_r_out = None
        self.deriv_wrt_t_out = None

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
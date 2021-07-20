import numpy as np


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
    def deriv_thickness_inner_wrt_r_out(self):
        volume = self.base_micelle.volume
        r_in = self.base_micelle.radius_inner
        g_in = self.base_micelle.surfactants_number_innner
        d_g_in_wrt_r_out = self.deriv_surfactants_number_in_wrt_r_out

        aux = 3.0 * g_in * volume / (4.0 * np.pi) + r_in * r_in * r_in
        s_1 = 3.0 * aux * aux * (d_g_in_wrt_r_out + 2. * r _in * r_in)
        s_2 = -1. * d_g_in_wrt_r_out

        deriv = s_1 + s_2
        return deriv

    def deriv_thickness_inner_wrt_t_out(self):
        volume = self.base_micelle.volume
        r_in = self.base_micelle.radius_inner
        g_in = self.base_micelle.surfactants_number_innner
        d_g_in_wrt_t_out = self.deriv_surfactants_number_in_wrt_t_out

        aux = 3.0 * g_in * volume / (4.0 * np.pi) + r_in * r_in * r_in
        s_1 = 3.0 * aux * aux * (d_g_in_wrt_r_out + 2. * r _in * r_in)
        s_2 = -1. * d_g_in_wrt_t_out

        deriv = s_1 + s_2
        return deriv
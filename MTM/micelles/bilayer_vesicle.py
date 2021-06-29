from .base_micelle import BaseMicelle
import numpy as np
import warnings
from scipy.optimize import minimize


class BilayerVesicle(BaseMicelle):
    """
    Class for the calculation of chemical potential incentive of
    bilayer vesicle micelles. Follows Enders and Haentzschel 1998.
    """

    def __init__(self, *args, throw_errors=True):
        """
        Parameters
        ----------
        throw_errors : bool, optional
            wheter to throw an error or warning if optimisation of radii is not convergent, by default True
        """
        super().__init__(*args)
        self._r_out, self._t_out = self._starting_values()
        self.throw_errors = throw_errors

    @classmethod
    def optimised_radii(cls, *args, **kwargs):
        """
        Instantiate a BilayerVesicle instance with optimised outer radius
        and outer layer thickness.

        Returns
        -------
        object
            BilayerVesicle object
        """
        instance = cls(*args, **kwargs)
        instance.optimise_radii()
        return instance

    def optimise_radii(self, method="objective", hot_start=False):
        """
        Optimise outer radius and outer thickness of the vesicle to give
        lowest chemical potential.

        Parameters
        ----------
        method : str, optional
            'objective' for finding the radii via 
            an optimisation of the objective function, by default "objective"
        hot_start : bool, optional
            Use current values of outer radius and outer thickness.
            If false estimate from assumption of equal number of surfactans
            in outer/inner layer, by default False.

        Raises
        ------
        RuntimeError
            if optimisation is not totally successful 
            (if self.throw_error=False it's a warning)
        NotImplementedError
            if a method was chosen which is not implemented
        """
        if not hot_start:
            starting_values = np.asarray(self._starting_values())
        else:
            starting_values = np.asarray([self.radius_outer, self.thickness_outer])

        if method == "objective":
            Optim = minimize(
                self._optimiser_func, starting_values, bounds=((0, 10), (0, 10))
            )
        else:
            raise NotImplementedError("Method '{:s}' is not implemented".format(method))

        if not Optim.success:  # and Optim.status != 8:
            errmsg = "Error in Optimisation of micelle dimension: {:s}".format(
                str(Optim.message)
            )
            if self.throw_errors:
                raise RuntimeError(errmsg)
            warnings.warn(errmsg)
        else:
            self._r_out = Optim.x[0]
            self._t_out = Optim.x[1]

    def _optimiser_func(self, variables):
        """
        Update respective optimisation variables and calculate chemical
        potential difference.

        """
        self._r_out = variables[0]
        self._t_out = variables[1]
        obj_function = self.get_delta_chempot()
        if not self._check_geometry():
            obj_function = 10
        return obj_function

    def _starting_values(self):
        """
        Starting values for optimisation.

        Outer radius and thickness are set so that half of the surfactant
        is in the outer/inner layer and the inside surface area per surfactant 
        is 10% higher than the headgroup area.

        Returns
        -------
        list
            list of two floats, the outer radius and outer thickness.
        """
        number_surfactants = self.surfactants_number
        alkane_volume = number_surfactants * self.volume
        min_area_inner = 1.1 * self.headgroup_area * number_surfactants / 2.0
        r_inner = np.sqrt(min_area_inner / 4.0 / np.pi)
        r_outer = np.cbrt(alkane_volume / (4.0 / 3.0 * np.pi) + r_inner ** 3)

        r_middle = np.cbrt(r_outer ** 3 - alkane_volume / 2.0 / (4.0 / 3.0 * np.pi))
        t_outer = r_outer - r_middle

        return [r_outer, t_outer]

    def _check_geometry(self):
        if (
            self.surfactants_number_outer < 0
            or self.surfactants_number_inner < 0
            or self.radius_inner < 0
            or self.radius_outer < 0
            or self.thickness_inner < 0
            or self.thickness_outer < 0
        ):
            return False

    @property
    def radius_outer(self):
        """
        Returns
        -------
        float
            Outer radius of bilayer vesicle
        """
        return self._r_out

    @property
    def thickness_outer(self):
        """
        Returns
        -------
        float
            Outer thickness of the bilayer vesicle
        """
        return self._t_out

    @property
    def radius_inner(self):
        """
        Determined via the volume needed and the given outer radius.

        Returns
        -------
        float
            Inner radius of bilayer vesicle
        """
        _volume = self.volume
        _nr_surfactants = self.surfactants_number
        _r_out = self.radius_outer

        nom = 3.0 * _nr_surfactants * _volume
        denom = 4.0 * np.pi

        r_inner = np.cbrt(_r_out ** 3 - nom / denom)
        return r_inner

    @property
    def surfactants_number_outer(self):
        """
        Given by the volume between sphere with outer radius and outer
        radius minus outer thickness.

        Returns
        -------
        float
            Number of surfactants in outer layer of the bilayer vesicle.
        """
        _r_out = self.radius_outer
        _t_out = self.thickness_outer
        _volume = self.volume

        nom = 4.0 * np.pi * (_r_out ** 3 - (_r_out - _t_out) ** 3)
        denom = 3.0 * _volume

        g_out = nom / denom

        return g_out

    @property
    def surfactants_number_inner(self):
        """
        Returns
        -------
        float
            Number of surfactants in inner layer of the bilayer vesicle.
        """
        _nr_surfactants = self.surfactants_number
        _g_out = self.surfactants_number_outer

        return _nr_surfactants - _g_out

    @property
    def thickness_inner(self):
        """
        Determined via the volume needed for the inner layer surfactants and the inner radius.

        Returns
        -------
        float
            thickness of inner layer of bilayer vesicle
        """
        _r_in = self.radius_inner
        _volume = self.volume
        _g_in = self.surfactants_number_inner

        _ti = np.cbrt(_r_in ** 3 + (3.0 * _g_in * _volume / (4.0 * np.pi)))
        _ti -= _r_in

        return _ti

    @property
    def area_per_surfactant_outer(self):
        """
        Trivially calculated from outer radius

        Returns
        -------
        float
            area per surfactant in outer shell
        """
        _r_out = self.radius_outer
        _g_out = self.surfactants_number_outer

        area = 4.0 * np.pi * _r_out * _r_out / _g_out

        return area

    @property
    def area_per_surfactant_inner(self):
        """
        Trivially calculated from inner radius

        Returns
        -------
        float
            area per surfactant in inner shell
        """
        _r_in = self.radius_inner
        _g_in = self.surfactants_number_inner

        area = 4.0 * np.pi * _r_in * _r_in / _g_in

        return area

    @property
    def area_per_surfactant(self):
        """
        Area per surfactant. Weighted average over inner and outer layer.

        Returns
        -------
        float
            Area per surfactant, averaged.
        """
        _surfactants_number = self.surfactants_number
        _area_inner = self.area_per_surfactant_inner
        _g_in = self.surfactants_number_inner
        _g_out = self.surfactants_number_outer
        _area_outer = self.area_per_surfactant_outer

        area_inner = _area_inner * _g_in / _surfactants_number
        area_outer = _area_outer * _g_out / _surfactants_number

        area = area_inner + area_outer

        return area

    def _deformation_nagarajan(self):
        """
        Deformation free energy:

        outer layer: d_mu = 10 * pi^2 t_o^2 / (320. * N * L^2)
        inner layer: d_mu = 10 * pi^2 t_^2 / (160. * N * L^2)

        with N = length_alkane / L and L: segment length.
        Returns
        -------
        float
            surfactant weighted deformation free energy.
        """
        _surfactant_number = self.surfactants_number
        _g_out = self.surfactants_number_outer
        factor_out = _g_out / _surfactant_number
        factor_in = 1.0 - factor_out
        deformation_in = self._deformation_nagarajan_in()
        deformation_out = self._deformation_nagarajan_out()

        return factor_in * deformation_in + factor_out * deformation_out

    def _deformation_nagarajan_out(self):
        """
        Deformation free energy:

        outer layer: d_mu = 10 * pi^2 t_o^2 / (320. * N * L^2)

        with N = length_alkane / L and L: segment length.
        Returns
        -------
        float
            surfactant weighted deformation free energy.
        """

        _t_out = self.thickness_outer
        _length = self.length
        _segment_length = self.segment_length

        nom = 10.0 * np.pi * np.pi * _t_out * _t_out
        denom = 320.0 * _length * _segment_length
        deform = nom / denom

        return deform

    def _deformation_nagarajan_in(self):
        """
        Deformation free energy:

        inner layer: d_mu = 10 * pi^2 t_^2 / (160. * N * L^2)

        with N = length_alkane / L and L: segment length.
        Returns
        -------
        float
            surfactant weighted deformation free energy.
        """
        _t_in = self.thickness_inner
        _length = self.length
        _segment_length = self.segment_length

        nom = 10.0 * np.pi * np.pi * _t_in * _t_in
        denom = 160.0 * _length * _segment_length
        deform = nom / denom

        return deform

    def _steric_vdw(self):
        """
        Surfactant weighted steric interaction of headgroups:

        d_mu = \sum_j - g_j/g ln(1 - a_p / a_j)

        where j : {inner, outer} and a_j is the area per headgroup
        """

        def aux(a_head, a_avail, g_j, g):
            _ln = np.log(1.0 - a_head / a_avail)
            return -1.0 * g_j * _ln / g

        _surfactants_number = self.surfactants_number

        _headgroup_area = self.headgroup_area
        _area_inner = self.area_per_surfactant_inner
        _g_in = self.surfactants_number_inner
        if _headgroup_area > _area_inner:
            _steric_in = 10.0
        else:
            _steric_in = aux(_headgroup_area, _area_inner, _g_in, _surfactants_number)

        _g_out = self.surfactants_number_outer
        _area_outer = self.area_per_surfactant_outer

        if _area_outer < _headgroup_area or _area_inner < _headgroup_area:
            _steric_out = 10.0
        else:
            _steric_out = aux(_headgroup_area, _area_outer, _g_out, _surfactants_number)

        steric = _steric_in + _steric_out

        return steric

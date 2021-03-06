from .base_micelle import BaseMicelle
import numpy as np
import warnings
from scipy.optimize import minimize, LinearConstraint, root
from ._rodlike_derivative import RodlikeMicelleDerivative


class RodlikeMicelle(BaseMicelle):
    """
    Class for the calculation of chemical potential incentive of
    rodlike micelles.

    High level functionality implemented in BaseMicelle class -- see
    main methods there.
    """

    def __init__(self, *args, throw_errors=False):
        super().__init__(*args)
        self._r_sph, self._r_cyl = self._starting_values()
        self.throw_errors = throw_errors

    @classmethod
    def optimised_radii(cls, *args, **kwargs):
        """
        Return an instance with the radii already optmised via
        self.optimise_radii()

        Returns
        -------
        object
            instance of RodlikeMicelle class.
        """
        instance = cls(*args, **kwargs)
        instance.optimise_radii()
        return instance

    def optimise_radii(self, method="derivative", hot_start=True, x_0=None):
        """
        Optimise the radii of the rodlike micelle parts for minimal
        chemical potential difference.

        Parameters
        ----------
        method : str, optional
            'derivative' for finding the radii via a root search of
            the jacobian and 'objective' for finding the radii via
            an optimisation of the objective function, by default "derivative"
        hot_start : bool, optional
            Use current values of radius of spherical endcaps and radius
            of cylinder for starting values, by default True
        x_0 : list, optional
            starting values for radius of spherical endcaps and radius
            of cylinder, by default [1.3, 1.0]

        Raises
        ------
        RuntimeError
            if optimisation is not totally successful
            (if self.throw_error=False it's a warning)
        """
        Derivatives = RodlikeMicelleDerivative(self)
        if hot_start:
            x_0 = [self.radius_sphere, self.radius_cylinder]
        elif x_0 is not None:
            pass
        elif x_0 is None:
            x_0 = list(self._starting_values())

        if method == "derivative":
            Optim = root(
                Derivatives.jacobian, x0=x_0, method="hybr", options={"factor": 0.1}
            )
        elif method == "objective":
            constraints = {
                "type": "ineq",
                "fun": lambda x: np.array([x[0] - x[1] - 1e8]),
                "jac": lambda x: np.array([1.0, -1.0]),
            }

            upper_bound = 5
            Optim = minimize(
                self._optimiser_func,
                np.asarray(x_0),
                jac=Derivatives.jacobian,
                bounds=((0, upper_bound), (0, upper_bound)),
                constraints=constraints,
                options={"ftol": 1e-8},
            )
        else:
            raise RuntimeError("Method need either be derivative or objective")

        if not Optim.success:  # and Optim.status != 8:
            errmsg = "Error in Optimisation of micelle dimension: {:s}".format(
                Optim.message
            )
            if self.throw_errors:
                raise RuntimeError(errmsg)
            warnings.warn(errmsg)
        else:
            self._r_sph = Optim.x[0]
            self._r_cyl = Optim.x[1]

    def _optimiser_func(self, variables):
        self._r_sph = variables[0]
        self._r_cyl = variables[1]
        obj_function = self.get_delta_chempot()
        if self.geometry_check:
            obj_function = 10
        return obj_function

    def _starting_values(self):
        # rsph
        length = self.length
        r_sph = length * 0.9
        r_cyl = length * 0.7

        return r_sph, r_cyl

    @property
    def geometry_check(self):
        if (
            self.area_per_surfactant < 0
            or self.surfactants_number_cyl < 0
            or self.surfactants_number_cap < 0
            or self.area_per_surfactant_cyl < 0
            or self.area_per_surfactant_cap < 0
        ):
            return False
        return True

    @property
    def radius_sphere(self):
        return self._r_sph

    @property
    def radius_cylinder(self):
        return self._r_cyl

    @property
    def cap_height(self):
        """
        Also known as capital H in Enders 1998 or 'h' in
        http://www.ambrsoft.com/TrigoCalc/Sphere/Cap/SphereCap.htm
        There actually is a typo in Enders original work. The correct
        calculation is also given in Nagarajan 1991:

        H = r_s * (1 - sqrt(1-rc^2/rs^2))
        """
        _aux = self._r_cyl / self._r_sph
        return self._r_sph * (1.0 - np.sqrt(1.0 - (_aux * _aux)))

    @property
    def surfactants_number_cap(self):
        H = self.cap_height
        rs = self._r_sph
        Vs = self.volume

        aux = H * H * ((3.0 * rs) - H)
        nom = 2.0 * np.pi * ((4.0 * rs * rs * rs) - aux)
        denom = 3.0 * Vs

        return nom / denom

    @property
    def surfactants_number_cyl(self):
        return self.surfactants_number - self.surfactants_number_cap

    @property
    def cylinder_length(self):
        nom = self.volume * self.surfactants_number_cyl
        denom = self._r_cyl * self._r_cyl * np.pi

        return nom / denom

    @property
    def area_per_surfactant_cyl(self):
        return (
            2.0
            * np.pi
            * self._r_cyl
            * self.cylinder_length
            / self.surfactants_number_cyl
        )

    @property
    def area_per_surfactant_cap(self):
        h = self.cap_height

        return (
            4.0
            * np.pi
            * self._r_sph
            * ((2.0 * self._r_sph) - h)
            / self.surfactants_number_cap
        )

    @property
    def area_per_surfactant(self):
        a_cyl = self.area_per_surfactant_cyl
        a_cap = self.area_per_surfactant_cap
        g_cyl = self.surfactants_number_cyl
        g_cap = self.surfactants_number_cap

        area = (a_cyl * g_cyl) + (a_cap * g_cap)

        return area / (g_cap + g_cyl)

    def _deformation_nagarajan(self):
        _surfactants_number = self.surfactants_number
        _g_cap = self.surfactants_number_cap
        factor_cap = _g_cap / _surfactants_number
        factor_cyl = 1.0 - factor_cap
        deformation_cyl = self._deformation_nagarajan_cyl()
        deformation_sph = self._deformation_nagarajan_sph()

        return factor_cyl * deformation_cyl + factor_cap * deformation_sph

    def _deformation_nagarajan_cyl(self):
        _rc = self._r_cyl
        _length = self.length
        _segment_length = self.segment_length

        nom = 5.0 * np.pi * np.pi * _rc * _rc
        denom = 80.0 * _length * _segment_length

        return nom / denom

    def _deformation_nagarajan_sph(self):
        _rs = self._r_sph
        _length = self.length
        _segment_length = self.segment_length
        _volume = self.volume
        _a = self.area_per_surfactant_cap

        nom = 9.0 * _volume * np.pi * np.pi * _rs
        denom = 80.0 * _length * _segment_length * _a

        return nom / denom

    def _steric_vdw(self):
        _cap = self._steric_sph()
        _cyl = self._steric_cyl()

        return _cap + _cyl

    def _steric_sph(self):
        _headgroup_area = self.headgroup_area
        _area_per_surfactant = self.area_per_surfactant_cap
        _nr_surfactants = self.surfactants_number
        _nr_surfactants_portion = self.surfactants_number_cap
        if _headgroup_area >= _area_per_surfactant:
            return 10
            # raise ValueError(
            #    "headgroup area larger than area \
            #    per surfactant."
            # )
        else:
            ratio = _nr_surfactants_portion / _nr_surfactants
            return ratio * -np.log(1 - (_headgroup_area / _area_per_surfactant))

    def _steric_cyl(self):
        _headgroup_area = self.headgroup_area
        _area_per_surfactant = self.area_per_surfactant_cyl
        _nr_surfactants = self.surfactants_number
        _nr_surfactants_portion = self.surfactants_number_cyl
        if _headgroup_area >= _area_per_surfactant:
            return 10
            # raise ValueError(
            #    "headgroup area larger than area \
            #    per surfactant."
            # )
        else:
            ratio = _nr_surfactants_portion / _nr_surfactants
            return ratio * -np.log(1 - (_headgroup_area / _area_per_surfactant))

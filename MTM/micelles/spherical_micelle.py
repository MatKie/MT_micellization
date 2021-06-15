from .base_micelle import BaseMicelle
import numpy as np


class SphericalMicelle(BaseMicelle):
    """
    Class for the calculation of chemical potential incentive of 
    spherical micelles. 
    """

    @property
    def radius(self):
        """
        Radius of a sphere as given in Enders and Haentzschel 1998.
        """
        return np.cbrt(3 * self.surfactants_number * self.volume / (4.0 * np.pi))

    @property
    def area_per_surfactant(self):
        """
        Area per surfactant in a spherical micelle of given aggregation
        number and # of carbons in the tail and temperature.
        """
        radius = self.radius
        return 4.0 * np.pi * radius * radius / self.surfactants_number

    @property
    def area(self):
        """
        Area of a spherical micelle of given aggregation
        number and # of carbons in the tail and temperature.
        """
        return self.area_per_surfactant * self.surfactants_number

    def _deformation_nagarajan(self):
        _radius = self.radius
        _length = self.length
        _segment_length = self.segment_length

        nom = 9.0 * np.pi * np.pi * _radius * _radius
        denom = 240.0 * _length * _segment_length

        return nom / denom

    def _steric_vdw(self):
        _headgroup_area = self.headgroup_area
        _area_per_surfactant = self.area_per_surfactant
        if _headgroup_area >= _area_per_surfactant:
            raise ValueError(
                "headgroup area larger than area \
                per surfactant."
            )
        else:
            return -np.log(1 - (_headgroup_area / _area_per_surfactant))

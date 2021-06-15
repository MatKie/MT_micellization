from .base_micelle import BaseMicelle
import numpy as np


class GlobularMicelle(BaseMicelle):
    """
    Class for the calculation of chemical potential incentive of 
    globular micelles. 
    """

    def __init__(self, *args):
        super().__init__(*args)

    @property
    def area_per_surfactant(self):
        _length = self.length
        _volume = self.volume
        _g = self.surfactants_number

        vol_prolate = 4.0 / 3.0 * np.pi * _length * _length
        b = np.sqrt((_g * _volume) / vol_prolate)
        vol_prolate = 4.0 / 3.0 * np.pi * _length * _length * b
        _aux = _length / b
        E = np.sqrt(1.0 - (_aux * _aux))

        area = (
            2.0
            * np.pi
            * _length
            * _length
            * (1.0 + ((np.arcsin(E) * b) / (_length * E)))
            / _g
        )

        # area = 2.0 * np.pi * b * b * (1.0 + ((1.0 - (E * E)) / E) * np.arctanh(E)) / _g

        return area

    def _deformation_nagarajan(self):
        _volume = self.volume
        _area = self.area_per_surfactant
        _length = self.length
        _segment_length = self.segment_length

        nom = 9.0 * _volume * np.pi * np.pi * _length
        denom = 80.0 * _length * _segment_length * _area

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

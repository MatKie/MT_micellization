import pytest
import sys
import numpy as np
import os
from mkutils import create_fig, save_to_file
from MTM import literature
from MTM import MTSystem

sys.path.append("../")
this_path = os.path.dirname(__file__)


class TestInit:
    def test_init(self):
        MTS = MTSystem()
        ret = MTS.get_free_energy_minimas(g_0=80)
        assert ret is True

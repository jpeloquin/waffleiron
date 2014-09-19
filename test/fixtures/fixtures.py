# -*- coding: utf-8 -*-
import os
import unittest

import febtools as feb
from febtools.material import fromlame

class Hex8IsotropicCenterCrack(unittest.TestCase):
    """A 10 mm × 20 mm rectangle with a center 2 mm crack.

    2% strain applied in y.

    """
    def setUp(self):
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'center_crack_uniax_isotropic_elastic_hex8.feb'))
        self.model = reader.model()
        self.soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'center_crack_uniax_isotropic_elastic_hex8.xplt'))
        self.model.apply_solution(self.soln)

        material = self.model.mesh.elements[0].material
        y = material.y
        mu = material.mu
        E, nu = fromlame(y, mu)
        self.E = E
        self.nu = nu
        
        self.crack_line = ((-0.001, 0.0, 0.0),
                           ( 0.001, 0.0, 0.0))

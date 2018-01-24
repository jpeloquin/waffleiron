import os
import unittest

import numpy as np

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

        # right tip
        tip_line_r = [i for i, (x, y, z)
                      in enumerate(self.model.mesh.nodes)
                      if (np.allclose(x, self.crack_line[1][0])
                          and np.allclose(y, self.crack_line[1][1]))]
        self.tip_line_r = set(tip_line_r)

        tip_line_l = [i for i, (x, y, z)
                      in enumerate(self.model.mesh.nodes)
                      if (np.allclose(x, self.crack_line[0][0])
                          and np.allclose(y, self.crack_line[0][1]))]
        self.tip_line_l = set(tip_line_l)

        # identify crack faces
        f_candidates = feb.selection.surface_faces(self.model.mesh)
        f_seed = [f for f in f_candidates
                  if (len(set(f) & self.tip_line_r) > 1)]
        f_crack_surf = feb.selection.f_grow_to_edge(f_seed,
                                                    self.model.mesh)
        self.crack_faces = f_crack_surf

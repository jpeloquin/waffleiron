import febtools as feb
import unittest, os
import numpy as np
from numpy import dot
from math import degrees, radians, cos, sin

class QuadMesh(unittest.TestCase):
    """Quadrilateral mesh (2D case) with a hole.

    """
    
    def setUp(self):
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'uniax-2d-center-crack-1mm.feb'))
        self.model = reader.model()
        # rotate mesh to make the corner identification a little more
        # difficult
        a = radians(15)
        R = np.array([[cos(a), -sin(a), 0],
                      [sin(a), cos(a), 0],
                      [0, 0, 1]])
        nodes_new = [dot(R, node) for node in self.model.mesh.nodes]
        self.model.mesh.nodes = nodes_new
        self.model.mesh.update_elements()
            

    def test_select_corners(self):
        """Test for selection of four exterior corner nodes.

        """
        corner_nodes = feb.selection.corner_nodes(self.model.mesh)
        assert not set(corner_nodes) - set([0, 100, 5554, 5454])

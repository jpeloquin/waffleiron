# Run these tests with nose
import unittest
from copy import deepcopy
from febtools import Mesh

class MergeTestTri2d(unittest.TestCase):

    def setUp(self):
        """Create two meshes sharing a common boundary.

        """
        node = [(0.0, 0.0),
                (0.3, 0.0),
                (0.6, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0)]
        element = [(0, 1, 5),
                   (1, 4, 5),
                   (1, 2, 4),
                   (2, 3, 4)]
        self.mesh1 = Mesh(node, element)
        node = [(x, -1.0 * y) for (x, y) in node]
        self.mesh2 = Mesh(node, element)
    

    def test_mesh_merge_all(self):
        """Test merging two meshes, merging all common nodes.

        """
        mesh1 = deepcopy(self.mesh1)
        mesh2 = deepcopy(self.mesh2)
        mergemesh = mesh1.merge(mesh2)
        assert(len(mergemesh.element) == 8)
        assert(len(mergemesh.node) == 8)

    def test_mesh_merge_sel(self):
        """Test merging two meshes, merging only selected nodes.

        """
        pass
    

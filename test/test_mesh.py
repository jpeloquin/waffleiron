# Run these tests with pytest
import unittest
import waffleiron as wfl
from copy import deepcopy

### Merging meshes


class MergeTestTri2d(unittest.TestCase):
    def setUp(self):
        """Create two meshes sharing a common boundary."""
        nodes1 = [
            (0.0, 0.0),
            (0.3, 0.0),
            (0.6, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        nodes2 = [(x, -1.0 * y) for (x, y) in nodes1]
        elements = [(0, 1, 5), (1, 4, 5), (1, 2, 4), (2, 3, 4)]
        self.mesh1 = wfl.Mesh.from_ids(nodes1, elements, wfl.element.Tri3)
        self.mesh2 = wfl.Mesh.from_ids(nodes2, elements, wfl.element.Tri3)

    def test_mesh_merge_all(self):
        """Test merging two meshes, merging all common nodes."""
        mesh1 = deepcopy(self.mesh1)
        mesh2 = deepcopy(self.mesh2)
        mesh1.merge(mesh2)
        assert len(mesh1.elements) == 8
        assert len(mesh1.nodes) == 8

    def test_mesh_merge_sel(self):
        """Test merging two meshes, merging only selected nodes."""
        pass

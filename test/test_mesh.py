# -*- coding: utf-8 -*-
# Run these tests with nose
import unittest
import febtools as fbt
from copy import deepcopy
from febtools import Mesh

class MergeTestTri2d(unittest.TestCase):

    def setUp(self):
        """Create two meshes sharing a common boundary.

        """
        nodes1 = [(0.0, 0.0),
                (0.3, 0.0),
                (0.6, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0)]
        elements = [(0, 1, 5),
                   (1, 4, 5),
                   (1, 2, 4),
                   (2, 3, 4)]
        self.mesh1 = Mesh(nodes1, elements)
        nodes2 = [(x, -1.0 * y) for (x, y) in nodes1]
        self.mesh2 = Mesh(nodes2, elements)
    

    def test_mesh_merge_all(self):
        """Test merging two meshes, merging all common nodes.

        """
        mesh1 = deepcopy(self.mesh1)
        mesh2 = deepcopy(self.mesh2)
        mesh1.merge(mesh2)
        assert(len(mesh1.elements) == 8)
        assert(len(mesh1.nodes) == 8)

    def test_mesh_merge_sel(self):
        """Test merging two meshes, merging only selected nodes.

        """
        pass

### ElementContainingPoint

class ElementContainingPointHex8(unittest.TestCase):
    """Test looking up which element contains (x,y,z)

    """
    def setUp(self):
        # The cube is bounded by x ∈ [-1, 1], y ∈ [-1, 1],
        # and z ∈ [0, 2]
        self.soln = fbt.MeshSolution("test/fixtures/"
                                     "uniax-8cube.xplt")

    def test_outside(self):
        points = [(3, 3, 3),
             (-2.5, -3.0, -10),
             (-5.0, -3.0, 1.0)]
        for p in points:
            e = self.soln.element_containing_point(p)
            assert(e is None)

    def test_inside(self):
        e = self.soln.element_containing_point((-0.5, -0.5, 0.5))
        assert(e.eid == 0)
        e = self.soln.element_containing_point((-0.5, 0.5, 0.5))
        assert(e.eid == 2)

class ElementContainingPointQuad4(unittest.TestCase):
    """Test looking up with element contains a point (x, y, z)

    """
    def setUp(self):
        # The square is bounded by x ∈ [-1, 1], y ∈ [-1, 1]
        self.soln = fbt.MeshSolution("test/fixtures/"
                                     "uniax-quad4.xplt")

    def test_outside(self):
        points = [(3, 3, 0),
                  (-2.5, -3.0, 0),
                  (-5.0, -3.0, 0)]
        for p in points:
            e = self.soln.element_containing_point(p)
            assert(e is None)

    def test_inside(self):
        e = self.soln.element_containing_point((-0.5, -0.5, 0))
        assert(e.eid == 0)
        e = self.soln.element_containing_point((-0.5, 0.5, 0))
        assert(e.eid == 1)

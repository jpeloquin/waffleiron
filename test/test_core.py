"""Tests of core data structures.

Run these tests with pytest.

"""
import unittest, os
from copy import deepcopy

# Febtools modules
import febtools as feb
from febtools.select import elements_containing_point
from febtools import Mesh

### Face connectivity


class FaceConnectivityHex8(unittest.TestCase):
    """Test for correct face connectivity."""

    def setUp(self):
        reader = feb.input.FebReader(
            os.path.join("test", "fixtures", "center_crack_uniax_isotropic_elastic.feb")
        )
        self.mesh = reader.mesh()

    def test_edge(self):
        faces = self.mesh.faces_with_node(0)
        # Check total number of connected faces
        assert len(faces) == 6


### ElementContainingPoint


class ElementContainingPointCubeHex8(unittest.TestCase):
    """Test looking up which element contains (x,y,z)"""

    def setUp(self):
        # The cube is bounded by -1 ≤ x ≤ 1, -1 ≤ y ≤ 1, and 0 ≤ z ≤ 2
        self.model = feb.input.FebReader(
            os.path.join("test", "fixtures", "uniax-8cube.feb")
        ).model()
        self.bb = feb.core._e_bb(self.model.mesh.elements)

    def test_outside(self):
        points = [(3, 3, 3), (-2.5, -3.0, -10), (-5.0, -3.0, 1.0)]
        for p in points:
            elements = elements_containing_point(
                p, self.model.mesh.elements, bb=self.bb
            )
            assert not elements

    def test_inside(self):
        p = [(-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5)]

        elems = elements_containing_point(p[0], self.model.mesh.elements, bb=self.bb)
        assert len(elems) == 1
        assert elems[0] is self.model.mesh.elements[0]

        elems = elements_containing_point(p[1], self.model.mesh.elements, bb=self.bb)
        assert len(elems) == 1
        assert elems[0] is self.model.mesh.elements[2]


class ElementContainingPointNarrowParallelogram(unittest.TestCase):
    """Test point in element for narrow parallelograms.

    This test addresses the bug in which the elements which might
    contain point P were selected by finding the node N closest to P
    and checking if P was in any element connected to N.  However, N
    is not necessarily part of the element that contains P.

    """

    def setUp(self):
        nodes = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 10, 0),
            (0, 10, 0),
            (2, 5, 0),
            (2, 15, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 10, 1),
            (0, 10, 1),
            (2, 5, 1),
            (2, 15, 1),
        ]
        # a rectangle (on left)
        self.e1 = feb.element.Hex8.from_ids([0, 1, 2, 3, 6, 7, 8, 9], nodes)
        # a parallelogram (on right)
        self.e2 = feb.element.Hex8.from_ids([2, 4, 5, 3, 8, 10, 11, 9], nodes)
        self.mesh = feb.Mesh(nodes, [self.e1, self.e2])
        self.bb = feb.core._e_bb(self.mesh.elements)

    def test_point_in_rectangle(self):
        """Test points that are nearest e2 nodes, but in e1."""
        points = [(0.5, 0.5, 0.5), (0.9, 5, 0.5)]
        for p in points:
            elems = elements_containing_point(p, self.mesh.elements, bb=self.bb)
            assert len(elems) == 1
            assert elems[0] is self.e1


class ElementContainingPointQuad4(unittest.TestCase):

    """Test looking up with element contains a point (x, y, z)"""

    def setUp(self):
        # The square is bounded by x ∈ [-1, 1], y ∈ [-1, 1]
        self.soln = feb.input.XpltReader(
            os.path.join("test", "fixtures", "uniax-quad4.xplt")
        )
        self.model = feb.input.FebReader(
            os.path.join("test", "fixtures", "uniax-quad4.feb")
        ).model()
        self.model.apply_solution(self.soln)
        self.bb = feb.core._e_bb(self.model.mesh.elements)

    def test_outside(self):
        points = [(3, 3, 0), (-2.5, -3.0, 0), (-5.0, -3.0, 0)]
        for p in points:
            elements = elements_containing_point(
                p, self.model.mesh.elements, bb=self.bb
            )
            assert not elements

    def test_inside(self):
        e = elements_containing_point(
            (-0.5, -0.5, 0), self.model.mesh.elements, bb=self.bb
        )
        assert len(e) == 1
        assert e[0] is self.model.mesh.elements[0]

        e = elements_containing_point(
            (-0.5, 0.5, 0), self.model.mesh.elements, bb=self.bb
        )
        assert len(e) == 1
        assert e[0] is self.model.mesh.elements[1]

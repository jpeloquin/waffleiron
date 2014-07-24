# -*- coding: utf-8 -*-
# Run these tests with nose
import unittest, os
import febtools as feb
from febtools import Mesh
from copy import deepcopy

### Face connectivity

class FaceConnectivityHex8(unittest.TestCase):
    """Test for correct face connectivity.

    """
    def setUp(self):
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'center_crack_uniax_isotropic_elastic.feb'))
        self.mesh = reader.mesh()

    def test_edge(self):
        faces = self.mesh.faces_with_node[0]
        # Check total number of connected faces
        assert len(faces) == 6
        # Check full connections
        fc_faces = [f for f in faces if f.fc_faces]
        assert len(fc_faces) == 2
        f = fc_faces[0]
        assert len(f.ec_faces) == 16
        # Check edge connections
        ec_faces = [f for f in faces if not f.fc_faces]
        for f in ec_faces:
            assert len(f.ec_faces) == 10

### ElementContainingPoint

class ElementContainingPointHex8(unittest.TestCase):
    """Test looking up which element contains (x,y,z)

    """
    def setUp(self):
        # The cube is bounded by x ∈ [-1, 1], y ∈ [-1, 1],
        # and z ∈ [0, 2]
        self.soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'uniax-8cube.xplt'))
        self.model = feb.input.FebReader(os.path.join('test', 'fixtures', 'uniax-8cube.feb')).model()
        self.model.apply_solution(self.soln)

    def test_outside(self):
        points = [(3, 3, 3),
             (-2.5, -3.0, -10),
             (-5.0, -3.0, 1.0)]
        for p in points:
            e = self.model.mesh.element_containing_point(p)
            assert(e is None)

    def test_inside(self):
        e = self.model.mesh.element_containing_point((-0.5, -0.5, 0.5))
        assert(e is self.model.mesh.elements[0])
        e = self.model.mesh.element_containing_point((-0.5, 0.5, 0.5))
        assert(e is self.model.mesh.elements[2])

class ElementContainingPointQuad4(unittest.TestCase):
    """Test looking up with element contains a point (x, y, z)

    """
    def setUp(self):
        # The square is bounded by x ∈ [-1, 1], y ∈ [-1, 1]
        self.soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'uniax-quad4.xplt'))
        self.model = feb.input.FebReader(os.path.join('test', 'fixtures', 'uniax-quad4.feb')).model()
        self.model.apply_solution(self.soln)

    def test_outside(self):
        points = [(3, 3, 0),
                  (-2.5, -3.0, 0),
                  (-5.0, -3.0, 0)]
        for p in points:
            e = self.model.mesh.element_containing_point(p)
            assert(e is None)

    def test_inside(self):
        e = self.model.mesh.element_containing_point((-0.5, -0.5, 0))
        assert(e is self.model.mesh.elements[0])
        e = self.model.mesh.element_containing_point((-0.5, 0.5, 0))
        assert(e is self.model.mesh.elements[1])

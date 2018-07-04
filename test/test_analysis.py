# Run these tests with nose
from nose.tools import with_setup
import unittest
import os
import itertools, math
from math import pi, radians, cos, sin
import numpy.testing as npt
import numpy as np

import febtools as feb
from febtools.input import FebReader
from febtools.material import fromlame, tolame
from febtools.analysis import *
from febtools import material

from fixtures import Hex8IsotropicCenterCrack

class CenterCrackHex8(Hex8IsotropicCenterCrack):
    """Center cracked isotropic elastic plate in 3d.

    """

    def _griffith(self):
        """Calculate Griffith strain energy.

        """
        a = 1.0e-3 # m
        minima = np.min(self.model.mesh.nodes, axis=0)
        maxima = np.max(self.model.mesh.nodes, axis=0)
        width = maxima[0] - minima[0]
        elems_up_down = [e for e in self.model.mesh.elements
                         if (np.any(e.nodes[:,1] == minima[1]) or
                             np.any(e.nodes[:,1] == maxima[1]))]
        pavg = np.mean([e.material.tstress(e.f((0, 0, 0)))
                        for e in elems_up_down], axis=0)

        # Define G for plane stress
        K_I = pavg[1][1] * (pi * a * 1.0 /
                            cos(pi * a / width))**0.5
        G = K_I**2.0 / self.E
        return G

    def test_right_tip(self):
        # Calculate J

        zslice = feb.selection.element_slice(self.model.mesh.elements,
                                             v=0e-3, axis=(0, 0, 1))
        nodes = [n for e in zslice for n in e.nodes]
        maxima = np.max(nodes, axis=0)
        minima = np.min(nodes, axis=0)
        deltaL = maxima[2] - minima[2]

        # define integration domain
        domain = [e for e in zslice
                  if self.tip_line_r.intersection(e.ids)]
        domain = feb.selection.e_grow(domain, zslice, n=2)
        domain = feb.analysis.apply_q_3d(domain, self.crack_faces,
                                         self.tip_line_r,
                                         q=[1, 0, 0])
        #assert len(domain) == 4**2.0 * 4**2.0 * 2

        jbdl = feb.analysis.jintegral(domain)
        jbar = jbdl / (0.5 * deltaL)
        # 0.5 * deltaL is standing in for ∫q(η)dη; this is ok for a
        # tringular q(η)

        # Test if approximately equal to G
        G = self._griffith()
        npt.assert_allclose(jbar, G, rtol=0.07)
        # Test for consistency with value calculated when code
        # initially verified
        npt.assert_allclose(jbar, 73.33, rtol=1e-4)

    def test_left_tip(self):
        """Test if J is valid for left crack tip.

        """
        zslice = feb.selection.element_slice(self.model.mesh.elements,
                                             v=0e-3, axis=(0, 0, 1))
        nodes = [n for e in zslice for n in e.nodes]
        maxima = np.max(nodes, axis=0)
        minima = np.min(nodes, axis=0)
        deltaL = maxima[2] - minima[2]

        # define integration domain
        domain = [e for e in zslice
                  if self.tip_line_l.intersection(e.ids)]
        domain = feb.selection.e_grow(domain, zslice, n=2)
        domain = feb.analysis.apply_q_3d(domain, self.crack_faces,
                                         self.tip_line_l,
                                         q=[-1, 0, 0])
        assert len(domain) == 6 * 6 * 2

        jbdl = feb.analysis.jintegral(domain)
        jbar = jbdl / (0.5 * deltaL)
        # 0.5 * deltaL is standing in for ∫q(η)dη; this is ok for a
        # tringular q(η)

        # Test if approximately equal to G
        G = self._griffith()
        npt.assert_allclose(jbar, G, rtol=0.07)
        # Test for consistency with value calculated when code
        # initially verified
        npt.assert_allclose(jbar, 73.33, rtol=1e-4)

    def test_rotated_right_tip(self):
        """Test if J is the same after a coordinate shift.

        """
        mesh = self.model.mesh
        G = self._griffith()

        # Create object transform (rotation matrix)
        angle = radians(30)
        R = np.array([[cos(angle), -sin(angle), 0],
                      [sin(angle), cos(angle), 0],
                      [0, 0, 1]])
        # Transform nodes
        mesh.nodes = [np.dot(R, n) for n in mesh.nodes]
        mesh.update_elements()
        # Transform displacements
        for e in mesh.elements:
            e.properties['displacement'] = \
                np.dot(R, e.properties['displacement'].T).T

        # Calculate J
        zslice = feb.selection.element_slice(self.model.mesh.elements,
                                             v=0e-3, axis=(0, 0, 1))
        nodes = [n for e in zslice for n in e.nodes]
        deltaL = np.max(nodes, axis=0)[2] - np.min(nodes, axis=0)[2]

        # Right tip
        domain = [e for e in zslice
                  if self.tip_line_r.intersection(e.ids)]
        domain = feb.selection.e_grow(domain, zslice, n=2)
        domain = feb.analysis.apply_q_3d(domain, self.crack_faces,
                                         self.tip_line_r,
                                         q=[cos(angle), sin(angle), 0])
        assert len(domain) == 6 * 6 * 2

        jbar_r = feb.analysis.jintegral(domain) / (0.5 * deltaL)
        npt.assert_allclose(jbar_r, G, rtol=0.07)
        # Test for consistency with value calculated when code
        # initially verified
        npt.assert_allclose(jbar_r, 76.20, rtol=1e-4)

    def test_rotated_left_tip(self):
        """Test if J is the same after a coordinate shift.

        """
        mesh = self.model.mesh
        G = self._griffith()
        # Geometry (easier before rotation)

        # Create object transform (rotation matrix)
        angle = radians(30)
        R = np.array([[cos(angle), -sin(angle), 0],
                      [sin(angle), cos(angle), 0],
                      [0, 0, 1]])
        # Transform nodes
        mesh.nodes = [np.dot(R, n) for n in mesh.nodes]
        mesh.update_elements()
        # Transform displacements
        for e in mesh.elements:
            e.properties['displacement'] = \
                np.dot(R, e.properties['displacement'].T).T

        # Calculate J
        zslice = feb.selection.element_slice(self.model.mesh.elements,
                                             v=0e-3, axis=(0, 0, 1))

        nodes = [n for e in zslice for n in e.nodes]
        deltaL = np.max(nodes, axis=0)[2] - np.min(nodes, axis=0)[2]

        # define integration domain
                # Right tip
        domain = [e for e in zslice
                  if self.tip_line_l.intersection(e.ids)]
        domain = feb.selection.e_grow(domain, zslice, n=2)
        domain = feb.analysis.apply_q_3d(domain, self.crack_faces,
                                         self.tip_line_l,
                                         q=[-cos(angle), -sin(angle), 0])
        assert len(domain) == 6 * 6 * 2

        jbar_l = feb.analysis.jintegral(domain) / (0.5 * deltaL)
        npt.assert_allclose(jbar_l, G, rtol=0.07)
        # Test for consistency with value calculated when code
        # initially verified
        npt.assert_allclose(jbar_l, 76.20, rtol=1e-4)


class CenterCrackQuad4(unittest.TestCase):

    def setUp(self):
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'center_crack_uniax_isotropic_elastic_quad4.feb'))
        self.model = reader.model()
        self.soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'center_crack_uniax_isotropic_elastic_quad4.xplt'))
        self.model.apply_solution(self.soln)

        material = self.model.mesh.elements[0].material
        y = material.y
        mu = material.mu
        E, nu = fromlame(y, mu)
        self.E = E
        self.nu = nu

    def test_jintegral(self):
        """Test j integral for Quad4 mesh, isotropic elastic material, small strain.

        """
        a = 1.0e-3 # m
        W = 10.0e-3 # m
        minima = np.min(self.model.mesh.nodes, axis=0)
        maxima = np.max(self.model.mesh.nodes, axis=0)
        ymin = minima[1]
        ymax = maxima[1]

        def pk1(element_ids):
            """Convert Cauchy stress in each element to 1st P-K.

            """
            data = self.soln.step_data()
            for i in element_ids:
                t = data['element variables']['stress'][i]
                e = self.model.mesh.elements[i]
                f = e.f((0,0))
                fdet = np.linalg.det(f)
                finv = np.linalg.inv(f)
                P = fdet * np.dot(finv, np.dot(t, finv.T))
                yield P

        e_top = [(i, e) for i, e
                 in enumerate(self.model.mesh.elements)
                 if np.any(np.isclose(e.nodes[:,1], ymin))]
        e_bot = [(i, e) for i, e
                 in enumerate(self.model.mesh.elements)
                 if np.any(np.isclose(e.nodes[:,1], ymax))]
        P = list(pk1([i for i, e in e_top + e_bot]))
        Pavg = sum(P) / len(P)
        stress = Pavg[1][1]

        # calculate stress intensity
        K_I = stress * (math.pi * a * 1.0 /
                        math.cos(math.pi * a / W))**0.5
        # Felderson; accurate to 0.3% for a/W ≤ 0.35
        G = K_I**2.0 / self.E

        id_crack_tip = [self.model.mesh.find_nearest_nodes(*(1e-3, 0.0, 0.0))[0]]
        elements = apply_q_2d(self.model.mesh, id_crack_tip, n=2,
                              q=[1, 0, 0])
        J = jintegral(elements)
        npt.assert_allclose(J, 72.75, atol=0.01)

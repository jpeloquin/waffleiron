# -*- coding: utf-8 -*-
# Run these tests with nose
from nose.tools import with_setup
import unittest
import os
import itertools, math
import numpy.testing as npt
import numpy as np

import febtools as feb
from febtools.input import FebReader
from febtools.material import fromlame, tolame
from febtools.analysis import *
from febtools import material

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def plot_q(elements):
    """Plot q function from mesh.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # get list of q values
    nodes = np.array([x for e in elements for x in e.nodes])
    n = nodes.shape[0]
    q = np.array([v for e in elements for v in e.properties['q']])
    # plot 1-values
    xyz = nodes[np.any(q, axis=1)]
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
               s=16, c='b', marker='o', edgecolor='b')
    # plot 0-values
    xyz = nodes[~np.any(q, axis=1)]
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
               s=16, c='r', marker='*', edgecolor='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax


class temp():

    def set_up_center_crack_2d_iso():
        soln = febtools.input.XpltReader(os.path.join('test', 'fixtures', 'center-crack-2d-1mm.xplt'))
        model = febtools.input.FebReader(os.path.join('test', 'fixtures', 'center-crack-2d-1mm.feb')).model()
        model.apply_solution(soln)

    @with_setup(set_up_center_crack_2d_iso)
    def test_select_elems_around_node():
        id_crack_tip = 1669
        elements = select_elems_around_node(model.mesh, id_crack_tip, n=2)
        expected = [1585, 1637, 1638, 1586, # ring 1
                    1534, 1535, 1587, 1639, # ring 2
                    1691, 1690, 1689, 1688,
                    1636, 1584, 1532, 1533]
        expected = set([model.mesh.elements[i] for i in expected])
        assert not elements - expected

    @with_setup(set_up_center_crack_2d_iso)
    def test_jdomain_q():
        id_crack_tip = 1669
        elements, q = jdomain(model.mesh, id_crack_tip, n=2, qtype='plateau')
        qexpected = [None] * len(model.mesh.nodes)
        i_inner = [1615, 1616, 1617, 1668, 1669, 1670,
                   1721, 1722, 1723, 2921]
        for i in i_inner:
            qexpected[i] = 1.0
        i_outer = [1561, 1562, 1563, 1564, 1565, 1614, 1618,
                   1667, 1671, 1720, 1724, 1773, 1774, 1775,
                   1776, 1777, 2920]
        for i in i_outer:
            qexpected[i] = 0.0
        npt.assert_array_equal(q, qexpected)


class CenterCrackHex8(unittest.TestCase):
    """Center cracked isotropic elastic plate in 3d.

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

    def test_jintegral_vs_griffith(self):
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
        K_I = pavg[1][1] * (math.pi * a * 1.0 /
                            math.cos(math.pi * a / width))**0.5
        G = K_I**2.0 / self.E

        # Calculate J
        tip_line = [i for i, (x, y, z)
                    in enumerate(self.model.mesh.nodes)
                    if np.allclose(x, 1e-3) and np.allclose(y, 0)]
        zslice = feb.selection.element_slice(self.model.mesh.elements,
                                             v=0e-3, axis=(0, 0, 1))
        nodes = [n for e in zslice for n in e.nodes]
        maxima = np.max(nodes, axis=0)
        minima = np.min(nodes, axis=0)
        deltaL = maxima[2] - minima[2]
        domain = feb.analysis.apply_q_3d(zslice, tip_line, n=3,
                                         q=[1, 0, 0])
        assert len(domain) == 6 * 6 * 2
        jbdl = feb.analysis.jintegral(domain)
        jbar = jbdl / (0.5 * deltaL)
        # 0.5 * deltaL is standing in for ∫q(η)dη; this is ok for a
        # tringular q(η)

        # debugging visualization
        plt.ion()
        fig, ax = plot_q(domain)
        plt.show()

        elems = [e for e in list(domain)
                 if np.any(np.array(e.nodes)[:,2] == minima[1])]
        # Test if approximately equal to G
        npt.assert_allclose(jbar, G, rtol=0.07)
        # Test for consistency with value calculated when code
        # initially verified
        npt.assert_allclose(jbar, 74.65, rtol=1e-4)


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

    def test_jintegral_vs_griffith(self):
        a = 1.0e-3 # m
        W = 10.0e-3 # m
        minima = np.min(self.model.mesh.nodes, axis=0)
        maxima = np.max(self.model.mesh.nodes, axis=0)
        ymin = minima[1]
        ymax = maxima[1]

        def pk1(element_ids):
            """Convert Cauchy stress in each element to 1st P-K.

            """
            data = self.soln.stepdata()
            for i in element_ids:
                t = data['element']['stress'][i]
                e = self.model.mesh.elements[i]
                f = e.f((0,0))
                fdet = np.linalg.det(f)
                finv = np.linalg.inv(f)
                P = fdet * np.dot(finv, np.dot(t, finv.T))
                yield P

        e_top = [(i, e) for i, e
                 in enumerate(self.model.mesh.elements)
                 if np.any(np.isclose(zip(*e.nodes)[1], ymin))]
        e_bot = [(i, e) for i, e
                 in enumerate(self.model.mesh.elements)
                 if np.any(np.isclose(zip(*e.nodes)[1], ymax))]
        P = list(pk1([i for i, e in e_top + e_bot]))
        Pavg = sum(P) / len(P)
        stress = Pavg[1][1]

        K_I = stress * (math.pi * a * 1.0 /
                        math.cos(math.pi * a / W))**0.5
        # Felderson; accurate to 0.3% for a/W ≤ 0.35
        G = K_I**2.0 / self.E
        id_crack_tip = [self.model.mesh.find_nearest_node(*(1e-3, 0.0, 0.0))]
        elements = apply_q_2d(self.model.mesh, id_crack_tip, n=3,
                              q=[1, 0, 0])
        J = jintegral(elements)
        npt.assert_allclose(J, G, rtol=0.03)

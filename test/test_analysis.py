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

# def test_jintegral():
"""J integral for isotropic material, equibiaxial stretch.

"""
febreader = FebReader(os.path.join('test', 'fixtures', 'center-crack-2d-1mm.feb'))
model = febreader.model()
materials = febreader.materials()
fp = os.path.join('test', 'fixtures', 'center-crack-2d-1mm.xplt')
soln = feb.input.XpltReader(fp)
y, mu = febtools.material.tolame(1e7, 0.3)
mat1 = {'type': 'isotropic elastic',
        'properties': {'lambda': y,
                       'mu': mu}}
m = {1: mat1}

x = (1e-3, 0)
id_crack_tip = model.mesh.find_nearest_node(*x)
elements, q = jdomain(model.mesh, id_crack_tip, n=2)
for e in elements:
    e.apply_property('q', [q[i] for i in e.ids])
j = jintegral(elements, q)

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

class CenterCrackQuad4(unittest.TestCase):

    def setUp(self):
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'uniax-2d-center-crack-1mm.feb'))
        self.model = reader.model()
        self.soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'uniax-2d-center-crack-1mm.xplt'))
        self.t = 0.2
        self.model.apply_solution(self.soln, t=self.t)

        material = self.model.mesh.elements[0].material
        y = materials[0].y
        mu = materials[0].mu
        E, nu = fromlame(y, mu)
        self.E = E
        self.nu = nu

    def test_jintegral_vs_griffith(self):
        a = 1.0e-3 # m
        W = 10.0e-3 # m
        minima = np.array([min(x) for x in zip(*model.mesh.nodes)])
        maxima = np.array([max(x) for x in zip(*model.mesh.nodes)])
        ymin = minima[1]
        ymax = maxima[1]
    
        def pk1(element_ids):
            """Convert Cauchy stress in each element to 1st P-K.
    
            """
            for i in element_ids:
                data = self.soln.stepdata(time=self.t)
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
        G = K_I**2.0 / self.E
        id_crack_tip = self.model.mesh.find_nearest_node(*(1e-3, 0.0, 0.0))
        elements, q = jdomain(self.model.mesh, id_crack_tip, n=3)
        for e in elements:
            e.apply_property('q', [q[i] for i in e.ids])
        J = jintegral(elements, q)
        npt.assert_allclose(J, G, rtol=0.01)

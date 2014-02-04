# Run these tests with nose
from nose.tools import with_setup
import unittest
import itertools, math
import numpy.testing as npt
import numpy as np

import febtools
from febtools.analysis import *
from febtools import material

# def test_jintegral():
"""J integral for isotropic material, equibiaxial stretch.

"""
f = 'test/j-integral/center-crack-2d-1mm.xplt'
y, mu = febtools.material.tolame(1e7, 0.3)
mat1 = {'type': 'isotropic elastic',
        'properties': {'lambda': y,
                       'mu': mu}}
m = {1: mat1}
soln = febtools.MeshSolution(f, matl_map=m)

x = (1e-3, 0)
id_crack_tip = soln.find_nearest_node(*x)
elements, q = jdomain(soln, id_crack_tip, n=2)
j = jintegral(elements, soln.data['displacement'],
              q, soln.material_map)

def set_up_center_crack_2d_iso():
    f = 'test/j-integral/center-crack-2d-1mm.xplt'
    y, mu = febtools.material.tolame(1e7, 0.3)
    mat1 = {'type': 'isotropic elastic',
            'properties': {'lambda': y,
                           'mu': mu}}
    m = {1: mat1}
    soln = febtools.MeshSolution(f, matl_map=m)

@with_setup(set_up_center_crack_2d_iso)
def test_select_elems_around_node():
    id_crack_tip = 1669
    elements = select_elems_around_node(soln, id_crack_tip, n=2)
    eid = [e.eid for e in elements].sort()
    expected = [1533, 1534, 1535, 1536,
                1585, 1586, 1587, 1588,
                1637, 1638, 1639, 1640,
                1689, 1690, 1691, 1692].sort()
    npt.assert_array_equal(eid, expected)

@with_setup(set_up_center_crack_2d_iso)
def test_jdomain_q():
    id_crack_tip = 1669
    elements, q = jdomain(soln, id_crack_tip, n=2, qtype='plateau')
    qexpected = [None] * len(soln.node)
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


def test_jintegral_uniax_center_crack_2d():
    E = 1e7
    nu = 0.3
    y, mu = febtools.material.tolame(E, nu)
    mat1 = {'type': 'isotropic elastic',
        'properties': {'lambda': y,
                       'mu': mu}}
    m = {1: mat1}
    soln = febtools.MeshSolution('test/fixtures/'
                                 'uniax-2d-center-crack-1mm.xplt',
                                 step=2,
                                 matl_map=m)
    a = 1.0e-3 # m
    W = 10.0e-3 # m
    minima = np.array([min(x) for x in zip(*soln.node)])
    maxima = np.array([max(x) for x in zip(*soln.node)])
    ymin = minima[1]
    ymax = maxima[1]
    e_top = [e for e in soln.element
             if np.any(np.isclose(zip(*e.xnode)[1], ymin))]
    e_bottom = [e for e in soln.element
                if np.any(np.isclose(zip(*e.xnode)[1], ymax))]
    def pk1(elements):
        """Convert Cauchy stress in each element to 1st P-K.

        """
        for e in elements:
            t = soln.data['stress'][e.eid]
            f = e.f((0,0), soln.data['displacement'])
            fdet = np.linalg.det(f)
            finv = np.linalg.inv(f)
            P = fdet * np.dot(finv, np.dot(t, finv.T))
            yield P
    P = list(pk1(e_top + e_bottom))
    Pavg = sum(P) / len(P)
    stress = Pavg[1][1]
    K_I = stress * (math.pi * a * 1.0 / math.cos(math.pi * a / W))**0.5
    G = K_I**2.0 / E
    id_crack_tip = soln.find_nearest_node(*(1e-3, 0.0, 0.0))
    elements, q = jdomain(soln, id_crack_tip, n=3)
    J = jintegral(elements, soln.data['displacement'],
                  q, soln.material_map)
    npt.assert_allclose(J, G, rtol=0.01)

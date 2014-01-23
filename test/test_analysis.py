# Run these tests with nose
import itertools, math
import numpy.testing as npt
import numpy as np

import febtools
from febtools.analysis import jintegral
from febtools import material

# def test_jintegral():
"""J integral for isotropic material, equibiaxial stretch.

"""
f = 'test/j-integral/center-crack-2d-1mm.xplt'
y, mu = febtools.material.IsotropicElastic.tolame(1e7, 0.3)
mat1 = {'type': 'isotropic elastic',
        'properties': {'lambda': y,
                       'mu': mu}}
m = {1: mat1}
soln = febtools.MeshSolution(f, matl_map=m)

x = (1e-3, 0)
id_crack_tip = soln.find_nearest_node(*x)
area1 = soln.elem_of_node(id_crack_tip)
area2 = soln.conn_elem(area1)
area = area2

# Nodes on the boundary will have different connectivity
def node_connectivity(elements, n):
    connectivity = [0] * n
    for e in elements:
        for i in e.inode:
            connectivity[i] += 1
    return connectivity

if isinstance(soln.element[0], febtools.element.Quad4):
    c_interior = 4
elif isinstance(soln.element[0], febtools.element.Hex8):
    c_interior = 8

c_mesh = node_connectivity(soln.element, len(soln.node))

domain = (soln.element[eid].inode for eid in area2)
domain = [nid for nid in itertools.chain.from_iterable(domain)] # flatten
domain = set(domain)

mesh_interior = set([i for i, c in enumerate(c_mesh)
                     if c == c_interior])
domain_crack = (domain - mesh_interior).union(set([id_crack_tip]))

c_domain = node_connectivity([soln.element[i] for i in area2],
                             len(soln.node))
domain_interior = set([i for i, c in enumerate(c_domain)
                       if c == c_interior and i != id_crack_tip])
domain_border = domain - domain_interior - domain_crack
domain_corners = set([i for i, c in enumerate(c_domain)
                      if c == 1])
domain_border = domain_border.union(domain_corners)

# define plateau function
q = [None] * len(soln.node)
for i in domain_border:
    q[i] = 0.0
for i in domain_crack:
    q[i] = 1.0
for i in domain_interior:
    q[i] = 1.0

domain_e = [soln.element[i] for i in area]
u = soln.data['displacement']
j = jintegral(domain_e, u, q, soln.material_map)

def test_jintegral_uniax_center_crack_2d():
    E = 1e7
    nu = 0.3
    y, mu = febtools.material.IsotropicElastic.tolame(E, nu)
    mat1 = {'type': 'isotropic elastic',
        'properties': {'lambda': y,
                       'mu': mu}}
    m = {1: mat1}
    soln = febtools.MeshSolution('test/fixtures/'
                                 'uniax-2d-center-crack-1mm.xplt',
                                 step=2,
                                 matl_map=m)
    a = 1.0 # mm
    W = 10.0 # mm
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


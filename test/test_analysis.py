# Run these tests with nose
import itertools
import numpy.testing as npt
import numpy as np

import febtools
from febtools.analysis import jintegral
from febtools import material

# def test_jintegral():
"""J integral for isotropic material, equibiaxial stretch.

"""
f = 'test/j-integral/center-crack-2d-1mm.xplt'
mat = {'Mat1': material.IsotropicElastic}
soln = febtools.MeshSolution(f, matl_map=mat)

x = (1e-3, 0)
id_crack_tip = soln.find_nearest_node(*x)
area1 = soln.elem_of_node(id_crack_tip)
area2 = soln.conn_elem(area1)

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

import pdb; pdb.set_trace()

# j = jintegral(area2, q)
#  return

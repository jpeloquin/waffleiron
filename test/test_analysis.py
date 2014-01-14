# Run these tests with nose

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
j = jintegral(soln, (1e-3, 0))
#  return

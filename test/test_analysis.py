# Run these tests with nose

import numpy.testing as npt
import numpy as np

import febtools
from febtools.analysis import jintegral

# def test_jintegral():
"""J integral for isotropic material, equibiaxial stretch.

"""
soln = febtools.MeshSolution('test/j-integral/'
                                 'center-crack-2d-1mm.xplt')
j = jintegral(soln, (1e-3, 0))
#  return

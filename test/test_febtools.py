# Run these tests with nose

import unittest
import numpy.testing as npt
import numpy as np
from nose.tools import with_setup
import febtools

"""Tests:
* Read xplt file, read log file, and compare
* Check calculated F tensor values against those read from log file
"""

print(dir(febtools))

class MeshSolutionTest(unittest.TestCase):
    xpltsol = febtools.MeshSolution('test/complex_loading.xplt')
    elemdata = febtools.readlog('test/complex_loading_elem_data.txt')
    nodedata = febtools.readlog('test/complex_loading_node_data.txt')

    def cmp_f(self, row, col, key):
        "Helper function for f tensor tests."
        for i, f in enumerate(self.xpltsol.f()):
            if np.isnan(self.elemdata[-1]['s1'][i]):
                # Likely a rigid body
                # FEBio gives rigid bodies F_ii = +/-1
                pass
            else:
                npt.assert_approx_equal(f[row, col], 
                                        self.elemdata[-1][key][i],
                                        significant=5)
    
    def test_fx(self):
        self.cmp_f(0, 0, 'Fx')

    def test_fy(self):
        self.cmp_f(1, 1, 'Fy')

    def test_fz(self):
        self.cmp_f(2, 2, 'Fz')

    def test_fxy(self):
        self.cmp_f(0, 1, 'Fxy')

    def test_fxz(self):
        self.cmp_f(0, 2, 'Fxz')

    def test_fyx(self):
        self.cmp_f(1, 0, 'Fyx')

    def test_fyz(self):
        self.cmp_f(1, 2, 'Fyz')

    def test_fzx(self):
        self.cmp_f(2, 0, 'Fzx')

    def test_fzy(self):
        self.cmp_f(2, 1, 'Fzy')

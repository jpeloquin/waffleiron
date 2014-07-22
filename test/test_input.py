# Run these tests with nose

import unittest
import os
import numpy.testing as npt
import numpy as np
from nose.tools import with_setup

import febtools

class MeshSolutionTest(unittest.TestCase):
    """Tests `MeshSolution.f`

    This test also depends on `readlog` and `Xpltreader` functioning
    correctly.  Results from an FEBio simulation are read from the
    text log and the binary file.  F tensors are computed for each
    element based on the binary file data and compared to those
    recorded in the text log.

    """

    def setUp(self):
        self.soln = febtools.input.XpltReader(os.path.join('test', 'fixtures', 'complex_loading.xplt'))
        reader = febtools.input.FebReader(os.path.join('test', 'fixtures', 'complex_loading.feb'))
        self.model = reader.model()
        self.model.apply_solution(self.soln
)
        self.elemdata = febtools.input.readlog(os.path.join('test', 'fixtures', 'complex_loading_elem_data.txt'))
        self.nodedata = febtools.input.readlog(os.path.join('test', 'fixtures', 'complex_loading_node_data.txt'))

    def cmp_f(self, row, col, key):
        """Helper function for comparing f tensors.

        Check the F tensor read from the xplt file against the text
        data in the logfile.

        """
        for i, e in enumerate(self.model.mesh.elements):
            # Check if rigid body. FEBio gives rigid bodies F_ii =
            # +/-1
            if not np.isnan(self.elemdata[-1]['s1'][i]):
                f = e.f((0, 0, 0))
                npt.assert_approx_equal(f[row, col], 
                                        self.elemdata[-1][key][i],
                                        significant=5)

    def test_fx(self):
        self.cmp_f(0, 0, 'Fxx')

    def test_fy(self):
        self.cmp_f(1, 1, 'Fyy')

    def test_fz(self):
        self.cmp_f(2, 2, 'Fzz')

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

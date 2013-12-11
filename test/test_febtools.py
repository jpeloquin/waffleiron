# Run these tests with nose

import unittest
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
    xpltsol = febtools.MeshSolution('test/complex_loading.xplt') 
    elemdata = febtools.readlog(
        'test/complex_loading_elem_data.txt') 
    nodedata = febtools.readlog(
        'test/complex_loading_node_data.txt')

    def cmp_f(self, row, col, key):
        """"Helper function for comparing f tensors.

        Check the F tensor read from the xplt file against the text
        data in the logfile.

        """
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


class MeshSolution1PKTest(unittest.TestCase):
    """Tests MeshSolution.s

    A known deformation gradient and cauchy stress is provided to
    `MeshSolution.s`, which calculates 1st Piola-Kirchoff stress
    $s$. The Cauchy stress is then recalculated based on $s$ and
    checked against the reference.

    """
    # Construct minimal instance of MeshSolution
    a = febtools.MeshSolution(None, -1)
    def f():
        f = []
        f.append(
            np.array([[ 1.17006749, -0.02775013,  0.02736336],
                      [ 0.07567887, -0.93846805, -0.13324592],
                      [-0.06877768,  0.13129845, -0.94069371]]))
        return f
    a.f = f
    a.data['stress'] = \
        [np.array([[ 45.576931  , -13.9562149 ,   2.88448954],
                   [-13.9562149 ,   8.23718643,  -0.69315594],
                   [  2.88448954,  -0.69315594,   4.93684435]])]
    print 'Setup completed'

    def test_cauchy_from_1pk(self):
        s = list(self.a.s())[0]
        f = self.a.f()[0]
        J = np.linalg.det(f)
        t_actual = 1 / J * np.dot(np.dot(f, s), f.T)
        t_desired = self.a.data['stress'][0]
        npt.assert_allclose(t_actual, t_desired)

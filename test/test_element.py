import febtools as feb
import numpy as np
import numpy.testing as npt
import unittest

def f_tensor_logfile(elemdata, step, eid):
    """Return F tensor read from a logfile.

    """
    Fxx = elemdata[step]['Fxx'][eid]
    Fyy = elemdata[step]['Fyy'][eid]
    Fzz = elemdata[step]['Fzz'][eid]
    Fxy = elemdata[step]['Fxy'][eid]
    Fxz = elemdata[step]['Fxz'][eid]
    Fyx = elemdata[step]['Fyx'][eid]
    Fyz = elemdata[step]['Fyz'][eid]
    Fzx = elemdata[step]['Fzx'][eid]
    Fzy = elemdata[step]['Fzy'][eid]
    F = np.array([[Fxx, Fxy, Fxz],
                  [Fyx, Fyy, Fyz],
                  [Fzx, Fzy, Fzz]])
    return F

def f_test_hex8():
    elemdata = feb.readlog('test/fixtures/'
                           'complex_loading_elem_data.txt')
    soln = feb.MeshSolution('test/fixtures/'
                            'complex_loading.xplt')
    istep = -1
    u = soln.reader.stepdata(istep)['displacement']
    for eid in xrange(len(soln.element) - 1): # don't check rigid body
        F_expected = f_tensor_logfile(elemdata, istep, eid)
        F = soln.element[eid].f((0, 0, 0), u)
        npt.assert_almost_equal(F, F_expected, decimal=5)


class FTestTri3(unittest.TestCase):
    """Test F tensor calculations for Tri3 mesh.

    Only part of the F tensor is tested right now, pending full
    implementation of the extended directors.

    """
    def setUp(self):
        self.soln = feb.MeshSolution('test/fixtures/'
                                     'square_tri3.xplt')
        self.elemdata = feb.readlog('test/fixtures/'
                                    'square_tri3_elem_data.txt')
    def test_f(self):
        istep = -1
        u = self.soln.reader.stepdata(istep)['displacement']
        for eid in xrange(len(self.soln.element)):
            F_expected = f_tensor_logfile(self.elemdata, istep, eid)
            F = self.soln.element[eid].f((1.0/3.0, 1.0/3.0), u)
            npt.assert_almost_equal(F[:2,:2], F_expected[:2,:2],
                                    decimal=5)


@unittest.skip("extended directors not yet implemented, so shell elements will not provide the correct F tensor")
class FTestQuad4(unittest.TestCase):
    """Test F tensor calculations for Tri3 mesh.

    """
    def setUp(self):
        self.soln = feb.MeshSolution('test/fixtures/'
                                     'square_quad4.xplt')
        self.elemdata = feb.readlog('test/fixtures/'
                                    'square_quad4_elem_data.txt')
    def test_f(self):
        istep = -1
        u = self.soln.reader.stepdata(istep)['displacement']
        for eid in xrange(len(self.soln.element)):
            F_expected = f_tensor_logfile(self.elemdata, istep, eid)
            F = self.soln.element[eid].f((1.0/3.0, 1.0/3.0), u)
            npt.assert_almost_equal(F[:2,:2], F_expected[:2,:2],
                                    decimal=5)

def test_integration():
    # create trapezoidal element
    node_list = ((0.0, 0.0),
                 (2.0, 0.0),
                 (1.5, 2.0),
                 (0.5, 2.0))
    nodes = (0, 1, 2, 3)
    element = feb.element.Quad4(nodes, node_list)
    # compute area
    actual = element.integrate(lambda e, r: 1.0)
    desired = 3.0 # A_trapezoid = 0.5 * (b1 + b2) * h
    npt.assert_approx_equal(actual, desired)

def test_dinterp():
    # create square element
    node_list = ((0.0, 0.0),
                 (1.0, 0.0),
                 (1.0, 1.0),
                 (0.0, 1.0))
    nodes = (0, 1, 2, 3)
    element = feb.element.Quad4(nodes, node_list)
    v = (0.0, 10.0, 11.0, 1.0)
    desired = np.array([10.0, 1.0])
    actual = element.dinterp((0,0), v).reshape(-1)
    npt.assert_allclose(actual, desired)

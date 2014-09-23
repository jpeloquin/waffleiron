import febtools as feb
import numpy as np
import numpy.testing as npt
import unittest, os 

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


class Hex8Element(unittest.TestCase):

    def setUp(self):
        nodes = [(-2, -1.5, -3),
                 (2, -1.5, -3),
                 (2, 1.0, -3),
                 (-2, 1.0, -3),
                 (-2, -1.5, 1.2),
                 (2, -1.5, 1.2),
                 (2, 1.0, 1.2),
                 (-2, 1.0, 1.2)]
        self.element = feb.element.Hex8(nodes)
        self.w = 4.0
        self.l = 2.5
        self.h = 4.2

    def test_j(self):
        desired = np.array([[self.w / 2, 0, 0],
                            [0, self.l / 2, 0],
                            [0, 0, self.h / 2]])
        # at center
        actual = self.element.j((0, 0, 0), config='reference')
        npt.assert_allclose(actual, desired)
        # at gauss points
        for pt in self.element.gloc:
            actual = self.element.j(pt, config='reference')
            npt.assert_allclose(actual, desired, atol=np.spacing(1))

    def test_dinterp_1d(self):
        dx = 1.5
        dy = -0.8
        dz = 0.7
        deltax = self.w * dx
        deltay = self.l * dy
        deltaz = self.h * dz
        self.element.properties['testval'] = np.array(
            (0.0, deltax,
             deltax + deltay, deltay,
             deltaz, deltax + deltaz,
             deltax + deltay + deltaz, deltay + deltaz))
        desired = np.array([dx, dy, dz])
        actual = self.element.dinterp((0, 0, 0), prop='testval')
        npt.assert_allclose(actual, desired)
        for pt in self.element.gloc:
            actual = self.element.dinterp(pt, prop='testval')
            npt.assert_allclose(actual, desired)

    def test_integration_volume(self):
        truth = self.w * self.l * self.h
        computed = self.element.integrate(lambda e, r: 1.0)
        npt.assert_approx_equal(computed, truth)

class ElementMethodsTestHex8(unittest.TestCase):

    def setUp(self):
        self.elemdata = feb.input.readlog(os.path.join('test', 'fixtures', 'complex_loading_elem_data.txt'))
        self.soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'complex_loading.xplt'))
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'complex_loading.feb'))
        self.model = reader.model()
        self.model.apply_solution(self.soln)

    def test_f(self):
        istep = -1
        u = self.soln.stepdata(istep)['node']['displacement']
        for eid in xrange(len(self.model.mesh.elements) - 1):
            # don't check rigid body (last element)
            F_expected = f_tensor_logfile(self.elemdata, istep, eid)
            F = self.model.mesh.elements[eid].f((0, 0, 0))
            npt.assert_almost_equal(F, F_expected, decimal=5)


class ElementMethodsTestQuad4(unittest.TestCase):

    def setUp(self):
        self.soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'center-crack-2d-1mm.xplt'))
        febreader = feb.input.FebReader(os.path.join('test', 'fixtures', 'center-crack-2d-1mm.feb'))
        self.model = febreader.model()
        self.model.apply_solution(self.soln)


class FTestTri3(unittest.TestCase):
    """Test F tensor calculations for Tri3 mesh.

    Only part of the F tensor is tested right now, pending full
    implementation of the extended directors.

    """
    def setUp(self):
        self.soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'square_tri3.xplt'))
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'square_tri3.feb'))
        self.model = reader.model()
        self.model.apply_solution(self.soln)
        self.elemdata = feb.input.readlog(os.path.join('test', 'fixtures', 'square_tri3_elem_data.txt'))

    def test_f(self):
        istep = -1
        u = self.soln.stepdata(step=istep)['node']['displacement']
        for eid in xrange(len(self.model.mesh.elements)):
            F_expected = f_tensor_logfile(self.elemdata, istep, eid)
            F = self.model.mesh.elements[eid].f((1.0/3.0, 1.0/3.0))
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
        for eid in xrange(len(self.soln.elements)):
            F_expected = f_tensor_logfile(self.elemdata, istep, eid)
            F = self.soln.elements[eid].f((1.0/3.0, 1.0/3.0), u)
            npt.assert_almost_equal(F[:2,:2], F_expected[:2,:2],
                                    decimal=5)

def test_integration():
    # create trapezoidal element
    nodes = ((0.0, 0.0),
             (2.0, 0.0),
             (1.5, 2.0),
             (0.5, 2.0))
    element = feb.element.Quad4(nodes)
    # compute area
    actual = element.integrate(lambda e, r: 1.0)
    desired = 3.0 # A_trapezoid = 0.5 * (b1 + b2) * h
    npt.assert_approx_equal(actual, desired)

def test_dinterp():
    # create square element
    nodes = ((0.0, 0.0),
             (1.0, 0.0),
             (1.0, 1.0),
             (0.0, 1.0))
    element = feb.element.Quad4(nodes)
    element.properties['testval'] = np.array((0.0, 10.0, 11.0, 1.0))
    desired = np.array([10.0, 1.0])
    actual = element.dinterp((0,0), 'testval').reshape(-1)
    npt.assert_allclose(actual, desired)

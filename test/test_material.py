# coding=utf-8

import unittest
import numpy as np
from numpy import dot
from numpy.linalg import inv
import numpy.testing as npt

import febtools as feb
from febtools.material import *
from febtools import FebReader


class ExponentialFiberTest(unittest.TestCase):
    """Tests exponential fiber material definition.

    Since this material is unstable, it must be tested in a mixture.
    The test, therefore, is not truly independent.

    """
    def setUp(self):
        soln = feb.MeshSolution('test/fixtures/'
                                'mixture_hm_exp.xplt')
        febreader = FebReader(open('test/fixtures/mixture_hm_exp.feb'))
        soln.assign_materials(febreader.materials())
        self.soln = soln

    def w_test(self):
        # This is a very weak test; just a sanity check.
        F = np.array([[1.1, 0.1, 0.0],
                      [0.2, 0.9, 0.0],
                      [-0.3, 0.0, 1.5]])
        matlprops = {'alpha': 65,
                     'beta': 2,
                     'ksi': 0.296,
                     'theta': 90,
                     'phi': 90}
        expfib = ExponentialFiber(matlprops)
        w = expfib.w(F)
        assert w > 0

    def tstress_test(self):
        """Check Cauchy stress against FEBio.

        """
        F = self.soln.elements[0].f((0, 0, 0),
                                   self.soln.data['displacement'])
        t_try = self.soln.elements[0].material.tstress(F)
        t_true = self.soln.data['stress'][0]
        npt.assert_allclose(t_true, t_try, rtol=1e-5)


class IsotropicElasticTest(unittest.TestCase):
    """Tests isotropic elastic material definition.

    """
    def setUp(self):
        elemdata = feb.readlog('test/fixtures/'
                              'isotropic_elastic_elem_data.txt')
        febreader = FebReader(open('test/fixtures/'
                                   'isotropic_elastic.feb'))
        materials = febreader.materials()
        self.material = materials[0]
        Fxx = elemdata[-1]['Fxx'][0]
        Fyy = elemdata[-1]['Fyy'][0]
        Fzz = elemdata[-1]['Fzz'][0]
        Fxy = elemdata[-1]['Fxy'][0]
        Fxz = elemdata[-1]['Fxz'][0]
        Fyx = elemdata[-1]['Fyx'][0]
        Fyz = elemdata[-1]['Fyz'][0]
        Fzx = elemdata[-1]['Fzx'][0]
        Fzy = elemdata[-1]['Fzy'][0]
        F = np.array([[Fxx, Fxy, Fxz],
                      [Fyx, Fyy, Fyz],
                      [Fzx, Fzy, Fzz]])
        self.elemdata = elemdata
        self.F = F

    def props_conversion_test(self):
        """Test IsotropicElastic creation from E and ν.

        """
        youngmod = 1e6
        nu = 0.4
        y, mu = tolame(youngmod, nu)
        matlprops = {'lambda': y,
                     'mu': mu}
        mat1 = IsotropicElastic(matlprops)
        mat2 = self.material
        w_try = mat1.w(self.F)
        w_true = mat2.w(self.F)
        npt.assert_approx_equal(w_try, w_true)

    def w_identity_test(self):
        F = np.eye(3)
        matlprops = {'lambda': 1.0,
                     'mu': 1.0}
        material = IsotropicElastic(matlprops)
        w_try = material.w(F)
        w_true = 0.0
        npt.assert_approx_equal(w_try, w_true)

    def w_test(self):
        F = np.array([[1.1, 0.1, 0.0],
                      [0.2, 0.9, 0.0],
                      [-0.3, 0.0, 1.5]])
        matlprops = {'lambda': 5.8e6,
                     'mu': 3.8e6}
        material = IsotropicElastic(matlprops)
        W_try = material.w(F)
        W_true = 3610887.5 # calculated by hand
        npt.assert_approx_equal(W_try, W_true)

    def sstress_test(self):
        tx = self.elemdata[-1]['sx'][0]
        ty = self.elemdata[-1]['sy'][0]
        tz = self.elemdata[-1]['sz'][0]
        txy = self.elemdata[-1]['sxy'][0]
        txz = self.elemdata[-1]['sxz'][0]
        tyz = self.elemdata[-1]['syz'][0]
        t_true = np.array([[tx, txy, txz],
                           [txy, ty, tyz],
                           [txz, tyz, tz]])
        Fdet = self.elemdata[-1]['J'][0]
        F = self.F
        s_true = Fdet * dot(inv(F), dot(t_true, inv(F.T)))
        s_try = self.material.sstress(F)
        npt.assert_allclose(s_try, s_true, rtol=1e-3, atol=1.0)

    def tstress_test(self):
        """Compare calculated stress with that from FEBio's logfile.

        """
        # someday, the material properties will be read from the .feb
        # file
        tx = self.elemdata[-1]['sx'][0]
        ty = self.elemdata[-1]['sy'][0]
        tz = self.elemdata[-1]['sz'][0]
        txy = self.elemdata[-1]['sxy'][0]
        txz = self.elemdata[-1]['sxz'][0]
        tyz = self.elemdata[-1]['syz'][0]
        t_true = np.array([[tx, txy, txz],
                           [txy, ty, tyz],
                           [txz, tyz, tz]])
        F = self.F
        t_try = self.material.tstress(F)
        npt.assert_allclose(t_try, t_true, rtol=1e-5)
        

class HolmesMowTestCase(unittest.TestCase):
    """Tests Holmes Mow material definition.

    """

    def setUp(self):
        soln = feb.MeshSolution('test/fixtures/holmes_mow.xplt')
        febreader = FebReader(open('test/fixtures/holmes_mow.feb'))
        soln.assign_materials(febreader.materials())
        self.soln = soln

    def tearDown(self):
        del self.soln

    def tstress_test(self):
        e = self.soln.elements[0]
        F = e.f((0, 0, 0), self.soln.data['displacement'])
        t_try = e.material.tstress(F)
        t_true = self.soln.data['stress'][0]
        npt.assert_allclose(t_try, t_true, rtol=1e-5)

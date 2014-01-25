import unittest
import numpy as np
from numpy import dot
from numpy.linalg import inv
import numpy.testing as npt

import febtools as feb
from febtools import material

class IsotropicElasticTest(unittest.TestCase):
    """Tests isotropic elastic material definition.

    """

    def setUp(self):
        elemdata = feb.readlog('test/fixtures/'
                              'isotropic_elastic_elem_data.txt')
        youngmod = 1e6
        nu = 0.4
        y, mu = material.IsotropicElastic.tolame(youngmod, nu)
        matlprops = {'lambda': y,
                     'mu': mu}
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
        self.matlprops = matlprops
        self.F = F

    def w_identity_test(self):
        F = np.eye(3)
        matlprops = {'lambda': 1.0,
                     'mu': 1.0}
        W_try = material.IsotropicElastic.w(F, self.matlprops)
        W_true = 0.0
        npt.assert_approx_equal(W_try, W_true)

    def w_test(self):
        F = np.array([[1.1, 0.1, 0.0],
                      [0.2, 0.9, 0.0],
                      [-0.3, 0.0, 1.5]])
        matlprops = {'lambda': 5.8e6,
                     'mu': 3.8e6}
        W_try = material.IsotropicElastic.w(F, matlprops)
        W_true = 3610887.5 # calculated by hand
        npt.assert_approx_equal(W_try, W_true)

    def s_test(self):
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
        S_true = Fdet * dot(inv(F), dot(t_true, inv(F.T)))
        S = material.IsotropicElastic.sstress(F, self.matlprops)
        npt.assert_allclose(S, S_true, rtol=1e-3, atol=1.0)

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
        t_try = material.IsotropicElastic.tstress(F, self.matlprops)
        npt.assert_allclose(t_try, t_true, rtol=1e-5)
        
        
    

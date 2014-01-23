import unittest
import numpy as np
import numpy.testing as npt

import febtools as fb
from febtools import material as mat

class IsotropicElasticTest(unittest.TestCase):
    """Tests isotropic elastic material definition.

    """
    def W_identity_test(self):
        F = np.eye(3)
        matlprops = {'lambda': 1.0,
                     'mu': 1.0}
        W_try = mat.IsotropicElastic.w(F, matlprops)
        W_true = 0.0
        npt.assert_approx_equal(W_try, W_true)

    def s_test(self):
        """Compare calculated stress with that from FEBio's logfile.

        """
        # someday, the material properties will be read from the .feb
        # file
        youngmod = 1e6
        nu = 0.4
        y, mu = mat.IsotropicElastic.tolame(youngmod, nu)
        matlprops = {'lambda': y,
                     'mu': mu}

        elemdata = fb.readlog('test/isotropic_elastic_elem_data.txt')
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
        tx = elemdata[-1]['sx'][0]
        ty = elemdata[-1]['sy'][0]
        tz = elemdata[-1]['sz'][0]
        txy = elemdata[-1]['sxy'][0]
        txz = elemdata[-1]['sxz'][0]
        tyz = elemdata[-1]['syz'][0]
        t_true = np.array([[tx, txy, txz],
                           [txy, ty, tyz],
                           [txz, tyz, tz]])
        t_try = mat.IsotropicElastic.tstress(F, matlprops)
        npt.assert_allclose(t_try, t_true, rtol=1e-5)
        
        
    

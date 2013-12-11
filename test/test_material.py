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
        y = 1.0
        mu = 1.0
        W_try = mat.IsotropicElastic.w(F, y, mu)
        W_true = 0.0
        npt.assert_approx_equal(W_try, W_true)

    def s_test(self):
        """Compare calculated stress with that from FEBio's logfile.

        """
        # someday, the material properties will be read from the feb
        # file
        youngmod = 1e6
        nu = 0.4
        y, mu = mat.IsotropicElastic.lameparam(youngmod, nu)

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
        sx = elemdata[-1]['sx'][0]
        sy = elemdata[-1]['sy'][0]
        sz = elemdata[-1]['sz'][0]
        sxy = elemdata[-1]['sxy'][0]
        sxz = elemdata[-1]['sxz'][0]
        syz = elemdata[-1]['syz'][0]
        s_true = np.array([[sx, sxy, sxz],
                           [sxy, sy, syz],
                           [sxz, syz, sz]])
        s_try = mat.IsotropicElastic.s(F, y, mu)
        npt.assert_allclose(s_try, s_true, rtol=1e-5)
        
        

import febtools as feb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import unittest
import os

import fixtures

#@unittest.skip("Very slow with bisection used to locate element containing a point.")
class ScalarFieldTest(unittest.TestCase):
    
    def setUp(self):
        soln = feb.input.XpltReader(os.path.join( \
            "test", "fixtures", "center_crack_uniax_isotropic_elastic_quad4.xplt"))
        febreader = feb.input.FebReader(os.path.join( \
            "test", "fixtures", "center_crack_uniax_isotropic_elastic_quad4.feb"))
        model = febreader.model()
        model.apply_solution(soln, t=1.0)
        self.model = model

    def test_y_stress(self):
        maxima = np.max(self.model.mesh.nodes, axis=0)
        minima = np.min(self.model.mesh.nodes, axis=0)
        px_w = 16
        scale = px_w / (maxima[0] - minima[0])
        px_h = (maxima[1] - minima[1]) * scale
        xi = np.linspace(minima[0], maxima[0], num=px_w)
        yi = np.linspace(minima[1], maxima[1], num=px_h)
        xv, yv = np.meshgrid(xi, yi)
        yv = np.flipud(yv)
        pts = np.concatenate([xv[...,np.newaxis],
                              yv[...,np.newaxis]], axis=2)
        # calculate y-stress
        fn = lambda f, e: e.material.tstress(f)[1,1]
        img = feb.plotting.scalar_field(self.model.mesh, fn, pts)

        fig = plt.figure()
        imgplot = plt.imshow(img)
        imgplot.set_interpolation('nearest')
        plt.ion()
        plt.savefig(os.path.join("test", "test_output", "scalar_field_test.png"))


class JDomainPlotTest(fixtures.Hex8IsotropicCenterCrack):
    """Test functions for visualizing J integral domain.

    """
    def test_plot_q(self):
        """Test 3D quiver plot of q vectors.

        """
        b = self.crack_line[1] # right crack tip
        tip_line = [i for i, (x, y, z)
                    in enumerate(self.model.mesh.nodes)
                    if np.allclose(x, b[0]) and np.allclose(y, b[1])]
        zslice = feb.selection.element_slice(self.model.mesh.elements,
            v=0.0, axis=np.array([0, 0, 1]))
        qdomain = feb.analysis.apply_q_3d(zslice, tip_line, n=5,
                                          q=np.array([1, 0, 0]))
        fig, ax = feb.plotting.plot_q(zslice, length=1e-4)
        fp_out = os.path.join("test", "test_output",
                              "jdomain_plot_test.png")
        fig.savefig(fp_out)

import febtools as feb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import unittest
import os

class ScalarFieldTest(unittest.TestCase):
    
    def setUp(self):
        soln = feb.input.XpltReader(os.path.join( \
            "test", "fixtures", "uniax-2d-center-crack-1mm.xplt"))
        febreader = feb.input.FebReader(os.path.join( \
            "test", "fixtures", "uniax-2d-center-crack-1mm.feb"))
        model = febreader.model()
        model.apply_solution(soln, t=1.0)
        self.model = model

    def test_y_stress(self):
        # TODO: changing px_w to 16 causes test failure

        maxima = np.max(self.model.mesh.nodes, axis=0)
        minima = np.min(self.model.mesh.nodes, axis=0)
        px_w = 32
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
        
        imgplot = plt.imshow(img)
        imgplot.set_interpolation('nearest')
        plt.ion()
        plt.show()
        plt.savefig(os.path.join("test", "test_output", "scalar_field_test.png"))
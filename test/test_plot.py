import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import unittest
import os

import pytest

import waffleiron as wfl
from waffleiron.test.fixtures import DIR_FIXTURES, DIR_OUT, gen_model_center_crack_Hex8


class ScalarFieldTest(unittest.TestCase):
    def setUp(self):
        self.model = wfl.load_model(
            DIR_FIXTURES / "center_crack_uniax_isotropic_elastic_quad4.xplt"
        )

    def test_y_stress(self):
        maxima = np.max(self.model.mesh.nodes, axis=0)
        minima = np.min(self.model.mesh.nodes, axis=0)
        px_w = 16
        scale = px_w / (maxima[0] - minima[0])
        px_h = (maxima[1] - minima[1]) * scale
        xi = np.linspace(minima[0], maxima[0], num=px_w)
        yi = np.linspace(minima[1], maxima[1], num=int(np.ceil(px_h)))
        xv, yv = np.meshgrid(xi, yi)
        yv = np.flipud(yv)
        pts = np.concatenate([xv[..., np.newaxis], yv[..., np.newaxis]], axis=2)
        # calculate y-stress
        fn = lambda f, e: e.material.tstress(f)[1, 1]
        img = wfl.plot.scalar_field(self.model.mesh, fn, pts)

        fig = plt.figure()
        imgplot = plt.imshow(img)
        imgplot.set_interpolation("nearest")
        fp = os.path.join("test", "output", "scalar_field_test.png")
        plt.savefig(fp)


@pytest.mark.skip(reason="too slow; hasn't changed in forever")
class JDomainPlotTest(unittest.TestCase):
    """Test functions for visualizing J integral domain."""

    def setUp(self):
        self.model, attrib = gen_model_center_crack_Hex8()
        self.crack_line = attrib["crack_line"]

    def test_plot_q(self):
        """Test 3D quiver plot of q vectors."""
        b = self.crack_line[1]  # right crack tip
        tip_line = [
            i
            for i, (x, y, z) in enumerate(self.model.mesh.nodes)
            if np.allclose(x, b[0]) and np.allclose(y, b[1])
        ]
        tip_line = set(tip_line)
        zslice = wfl.select.element_slice(
            self.model.mesh.elements, v=0.0, axis=np.array([0, 0, 1])
        )

        # find the crack faces
        candidates = wfl.select.surface_faces(self.model.mesh)
        f_seed = [f for f in candidates if (len(set(f) & set(tip_line)) > 1)]
        crack_faces = wfl.select.f_grow_to_edge(f_seed, self.model.mesh)

        qdomain = [e for e in self.model.mesh.elements if tip_line.intersection(e.ids)]
        qdomain = wfl.select.e_grow(qdomain, zslice, n=5)
        qdomain = wfl.analysis.apply_q_3d(
            zslice, crack_faces, tip_line, q=np.array([1, 0, 0])
        )
        fig, ax = wfl.plot.plot_q(zslice, length=1e-4)
        fig.savefig(DIR_OUT / "jdomain_plot_test.png")

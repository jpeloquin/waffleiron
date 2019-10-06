# Run these tests with nose
import unittest
import os
from pathlib import Path
import subprocess
import numpy.testing as npt
import numpy as np
from nose.tools import with_setup

import febtools as feb

class MeshSolutionTest(unittest.TestCase):
    """Tests `MeshSolution.f`

    This test also depends on `textdata_list` and `Xpltreader`
    functioning correctly.  Results from an FEBio simulation are read
    from the text log and the binary file.  F tensors are computed for
    each element based on the binary file data and compared to those
    recorded in the text log.

    """

    def setUp(self):
        self.soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'complex_loading.xplt'))
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'complex_loading.feb'))
        self.model = reader.model()
        self.model.apply_solution(self.soln)
        self.elemdata = feb.input.textdata_list(os.path.join('test', 'fixtures', 'complex_loading_elem_data.txt'), delim=",")
        self.nodedata = feb.input.textdata_list(os.path.join('test', 'fixtures', 'complex_loading_node_data.txt'), delim=",")

    def cmp_f(self, row, col, key):
        """Helper function for comparing f tensors.

        Check the F tensor read from the xplt file against the text
        data in the logfile.

        """
        for i, e in enumerate(self.model.mesh.elements):
            # Check if rigid body. FEBio gives rigid bodies F_ii =
            # +/-1
            if not np.isnan(self.elemdata[-1]['s1'][i]):
                f = e.f((0, 0, 0))
                npt.assert_approx_equal(f[row, col],
                                        self.elemdata[-1][key][i],
                                        significant=5)

    def test_fx(self):
        self.cmp_f(0, 0, 'Fxx')

    def test_fy(self):
        self.cmp_f(1, 1, 'Fyy')

    def test_fz(self):
        self.cmp_f(2, 2, 'Fzz')

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


class FebSolidFixedBCs(unittest.TestCase):
    """Test read of FEBio XML file with fixed boundary conditions.

    """
    path = Path("test") / "fixtures" / "cube_hex8_n=1_solid_all_BCs_fixed.feb"

    def setUp(self):
        subprocess.run(["febio", "-silent", "-i", str(self.path)])

    def test_load_model(self):
        model = feb.load_model(str(self.path))

    def test_read_mesh_from_xplt(self):
        pth_xplt = self.path.parent / f"{self.path.stem}.xplt"
        with open(str(pth_xplt), 'rb') as f:
            xplt = feb.xplt.XpltData(f.read())
        # xplt = feb.input.XpltReader(str(pth_xplt))
        mesh = xplt.mesh()
        # Check node values
        assert(len(mesh.nodes) == 8)
        assert(np.array(mesh.nodes).shape[1] == 3)
        npt.assert_almost_equal(mesh.nodes[3], [-0.5, 0.5, 0.0])
        # Check element values
        assert(len(mesh.elements) == 1)
        npt.assert_equal(mesh.elements[0].nodes, mesh.nodes)

    def tearDown(self):
        # Delete FEBio-generated output
        (self.path.parent / f"{self.path.stem}.log").unlink()
        (self.path.parent / f"{self.path.stem}.xplt").unlink()


class FebBiphasicFixedBCs(unittest.TestCase):
    """Test read of FEBio XML file with fixed boundary conditions.

    """
    path = Path("test") / "fixtures" / "cube_hex8_n=1_biphasic_all_BCs_fixed.feb"

    def setUp(self):
        subprocess.run(["febio", "-silent", "-i", str(self.path)])

    def test_load_model(self):
        model = feb.load_model(str(self.path))

    def test_read_mesh_from_xplt(self):
        pth_xplt = self.path.parent / f"{self.path.stem}.xplt"
        with open(str(pth_xplt), 'rb') as f:
            xplt = feb.xplt.XpltData(f.read())
        # xplt = feb.input.XpltReader(str(pth_xplt))
        mesh = xplt.mesh()
        # Check node values
        assert(len(mesh.nodes) == 8)
        assert(np.array(mesh.nodes).shape[1] == 3)
        npt.assert_almost_equal(mesh.nodes[3], [-0.5, 0.5, 0.0])
        # Check element values
        assert(len(mesh.elements) == 1)
        npt.assert_equal(mesh.elements[0].nodes, mesh.nodes)

    def tearDown(self):
        # Delete FEBio-generated output
        (self.path.parent / f"{self.path.stem}.log").unlink()
        (self.path.parent / f"{self.path.stem}.xplt").unlink()

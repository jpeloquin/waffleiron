import os
import unittest
from tvtk.api import tvtk
from tvtk.common import configure_input

import febtools as feb
from febtools.vtk import tvtk_ugrid_from_mesh


class ExportVTK(unittest.TestCase):

    def setUp(self):
        fp = os.path.join("test", "fixtures",
                          "center_crack_uniax_isotropic_elastic_hex8")
        self.model = feb.input.load_model(fp)

    def test_vtk_from_hex8(self):
        ugrid = tvtk_ugrid_from_mesh(self.model.mesh)

        fp_out = os.path.join("test", "output",
                              "unstructured_grid.vtu")
        if not os.path.exists(os.path.dirname(fp_out)):
            os.makedir(os.path.dirname(fp_out))
        w = tvtk.XMLUnstructuredGridWriter(file_name=fp_out)
        configure_input(w, ugrid)
        w.write()
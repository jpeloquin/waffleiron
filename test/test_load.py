# Python built-ins
from pathlib import Path
from unittest import TestCase
# Public modules
import numpy as np
import numpy.testing as npt
# febtools' local modules
import febtools as feb
from febtools.control import auto_control_section
from febtools.element import Hex8
from febtools.febioxml import basis_mat_axis_local
from febtools.model import Model, Mesh
from febtools.material import IsotropicElastic
from febtools.test.fixtures import gen_model_single_spiky_Hex8


DIR_THIS = Path(__file__).parent
DIR_FIXTURES = DIR_THIS / "fixtures"


def test_pipeline_prescribe_deformation_singleHex8():
    """Test conditions.prescribe_deformation()

    Test the following in the case of a displacement applied to all
    nodes in a single Hex8 element:

    - prescribe_deformation completes without error

    - the resulting model can be converted to FEBio XML

    - the resulting FEBio XML file can be solved by FEBio

    - the FEBio solution can be read by febtools

    - the F tensor returned by FEBio is the same as that which was input

    This is an integration test, not a unit test.  It is written as a
    single function because it tests the entire pipeline.

    """
    # Setup
    model = gen_model_single_spiky_Hex8()
    material = IsotropicElastic({"E": 10, "v": 0})
    for e in model.mesh.elements:
        e.material = material
    sequence = feb.Sequence(((0, 0), (1, 1)),
                            extend="extrapolate", interp="linear")
    model.add_step(control=auto_control_section(sequence, pts_per_segment=1))

    # Test 1: Does prescribe_deformation complete without error?
    F = np.array([[1.34, 0.18, -0.11],
                  [-0.20, 1.14, 0.17],
                  [-0.11, 0.20, 0.93]])
    node_set = feb.NodeSet([i for i in range(len(model.mesh.nodes))])
    feb.load.prescribe_deformation(model, node_set, F, sequence)

    # Test 2: Can the resulting model be converted to FEBio XML?
    fnm_stem = "prescribe_deformation_singleHex8"
    fnm_textdata = f"{fnm_stem}_-_element_data.txt"
    tree = feb.output.xml(model)
    e_Output = tree.find("Output")
    e_logfile = e_Output.makeelement("logfile")
    e_elementdata = e_logfile.makeelement("element_data",
                                          file=fnm_textdata,
                                          data="Fxx;Fyy;Fzz;Fxy;Fyz;Fxz;Fyx;Fzy;Fzx",
                                          format="%i %g %g %g %g %g %g %g %g %g")
    e_logfile.append(e_elementdata)
    e_Output.append(e_logfile)
    pth = DIR_THIS / "output" / f"{fnm_stem}.feb"
    with open(pth, "wb") as f:
        feb.output.write_xml(tree, f)

    # Test 3: Can FEBio use the resulting FEBio XML file?
    feb.febio.run_febio(pth)

    # Test 4: Can the FEBio solution be read by febtools?
    solved = feb.load_model(pth)

    # Test 5: Is the F tensor returned by FEBio the one that we expect?
    F_febio = feb.input.textdata_list(DIR_THIS / "output" / fnm_textdata)[-1]
    F_febio = np.array([[F_febio["Fxx"][0], F_febio["Fxy"][0], F_febio["Fxz"][0]],
                        [F_febio["Fyx"][0], F_febio["Fyy"][0], F_febio["Fyz"][0]],
                        [F_febio["Fzx"][0], F_febio["Fzy"][0], F_febio["Fzz"][0]]])
    npt.assert_array_almost_equal_nulp(F_febio, F)


def test_FEBio_prescribe_node_pressure_Hex8():
    """E2E test of prescribed nodal pressure boundary condition"""
    # Test 1: Read
    pth_in = DIR_FIXTURES / \
        (f"{Path(__file__).with_suffix('').name}." +\
         "prescribe_node_pressure.feb")
    model = feb.load_model(pth_in)
    #
    # Test 2: Write
    pth_out = DIR_THIS / "output" / \
        (f"{Path(__file__).with_suffix('').name}." +\
         "prescribe_node_pressure.feb")
    with open(pth_out, "wb") as f:
        feb.output.write_feb(model, f)
    # Test 3: Solve: Can FEBio use the roundtripped file?
    feb.febio.run_febio(pth_out)
    #
    # Test 4: Is the output as expected?
    model = feb.load_model(pth_out)
    e = model.mesh.elements[0]
    ##
    ## Test 4.1: Do we see the correct resultant fluid pressure?
    p_FEBio_1 = model.solution.value("fluid pressure", step=1,
                                     entity_id=0, region_id=1)
    npt.assert_almost_equal(p_FEBio_1, 50.0)
    p_FEBio_2 = model.solution.value("fluid pressure", step=2,
                                     entity_id=0, region_id=1)
    npt.assert_almost_equal(p_FEBio_2, 100.0)
    ##
    ## Test 4.2: Do we the correct resultant fluid flux?
    j_FEBio_1 = model.solution.value("fluid flux", step=1,
                                     entity_id=0, region_id=1)[0]
    j_FEBio_2 = model.solution.value("fluid flux", step=2,
                                     entity_id=0, region_id=1)[0]
    npt.assert_almost_equal(j_FEBio_1, 2.5)
    npt.assert_almost_equal(j_FEBio_2, 5.0)
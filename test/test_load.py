# Python built-ins
from pathlib import Path
from unittest import TestCase

# Public modules
import numpy as np
import numpy.testing as npt

# waffleiron' local modules
import waffleiron as wfl
from waffleiron import Step
from waffleiron.control import auto_ticker
from waffleiron.element import Hex8
from waffleiron.febioxml import basis_mat_axis_local
from waffleiron.model import Model, Mesh
from waffleiron.material import IsotropicElastic
from waffleiron.test.fixtures import DIR_OUT, febio_cmd_xml, gen_model_single_spiky_Hex8


DIR_THIS = Path(__file__).parent
DIR_FIXTURES = DIR_THIS / "fixtures"
DIR_OUTPUT = DIR_THIS / "output"


def test_pipeline_prescribe_deformation_singleHex8(febio_cmd_xml):
    """Test conditions.prescribe_deformation()

    Test the following in the case of a displacement applied to all
    nodes in a single Hex8 element:

    - prescribe_deformation completes without error

    - the resulting model can be converted to FEBio XML

    - the resulting FEBio XML file can be solved by FEBio

    - the FEBio solution can be read by waffleiron

    - the F tensor returned by FEBio is the same as that which was input

    This is an integration test, not a unit test.  It is written as a
    single function because it tests the entire pipeline.

    """
    febio_cmd, xml_version = febio_cmd_xml
    # Setup
    model = gen_model_single_spiky_Hex8()
    material = IsotropicElastic({"E": 10, "v": 0})
    for e in model.mesh.elements:
        e.material = material
    sequence = wfl.Sequence(((0, 0), (1, 1)), extrap="linear", interp="linear")
    step = Step(physics="solid", dynamics="static", ticker=auto_ticker(sequence))
    model.add_step(step)

    # Test 1: Does prescribe_deformation complete without error?
    F = np.array([[1.34, 0.18, -0.11], [-0.20, 1.14, 0.17], [-0.11, 0.20, 0.93]])
    node_set = wfl.NodeSet([i for i in range(len(model.mesh.nodes))])
    wfl.load.prescribe_deformation(model, step, node_set, F, sequence)

    # Test 2: Can the resulting model be converted to FEBio XML?
    fnm_stem = "prescribe_deformation_singleHex8"
    fnm_textdata = f"{fnm_stem}_-_element_data.txt"
    tree = wfl.output.xml(model, version=xml_version)
    e_Output = tree.find("Output")
    e_logfile = e_Output.makeelement("logfile")
    e_elementdata = e_logfile.makeelement(
        "element_data",
        file=fnm_textdata,
        data="Fxx;Fyy;Fzz;Fxy;Fyz;Fxz;Fyx;Fzy;Fzx",
        format="%i %g %g %g %g %g %g %g %g %g",
    )
    e_logfile.append(e_elementdata)
    e_Output.append(e_logfile)
    pth = DIR_OUT / f"{fnm_stem}.{febio_cmd}.xml{xml_version}.feb"
    with open(pth, "wb") as f:
        wfl.output.write_xml(tree, f)

    # Test 3: Can FEBio use the resulting FEBio XML file?
    wfl.febio.run_febio_checked(pth, cmd=febio_cmd)

    # Test 4: Can the FEBio solution be read by waffleiron?
    solved = wfl.load_model(pth)

    # Test 5: Is the F tensor returned by FEBio the one that we expect?
    F_febio = wfl.input.textdata_list(DIR_THIS / "output" / fnm_textdata)[-1]
    F_febio = np.array(
        [
            [F_febio["Fxx"][0], F_febio["Fxy"][0], F_febio["Fxz"][0]],
            [F_febio["Fyx"][0], F_febio["Fyy"][0], F_febio["Fyz"][0]],
            [F_febio["Fzx"][0], F_febio["Fzy"][0], F_febio["Fzz"][0]],
        ]
    )
    npt.assert_array_almost_equal_nulp(F_febio, F)


def test_FEBio_tied_elastic_contact_global(febio_cmd_xml):
    """E2E test of FEBio global tied elastic contact"""
    febio_cmd, xml_version = febio_cmd_xml
    # Test 1: Read
    mod = Path(__file__).with_suffix("").name
    pth_in = DIR_FIXTURES / f"{mod}.tied_elastic_contact_global.feb"
    model = wfl.load_model(pth_in)
    # Verify that contact exists
    assert len(model.constraints) == 1
    # Test 2: Write
    pth_out = DIR_OUTPUT / f"{pth_in.stem}.{febio_cmd}.xml{xml_version}.feb"
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, xml_version)
    # Test 3: Solve - Can FEBio use the roundtripped file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd)
    # Test 4: Is the output as expected?
    solved = wfl.load_model(pth_out)
    # Test 4.0: Did the nodes of the rigid indenter move down?
    rigid_nodes = [i for face in model.named["face sets"].obj("indenter") for i in face]
    for idx in rigid_nodes:
        δz = solved.solution.values("displacement", idx)["displacement"][-1][2]
        # Use loose tolerance b/c contact is imprecise
        npt.assert_allclose(δz, -0.11, rtol=5e-5)
    # Test 4.1: Did the top nodes of the deformable solid move down?
    top_nodes = [i for face in model.named["face sets"].obj("top") for i in face]
    for idx in top_nodes:
        δz = solved.solution.values("displacement", idx)["displacement"][-1][2]
        # Use loose tolerance b/c contact is imprecise
        npt.assert_allclose(δz, -0.11, rtol=5e-5)
    # Test 4.2: Was the correct strain produced?
    e_solid = [e for e in solved.mesh.elements if isinstance(e, wfl.element.Hex8)][0]
    F = e_solid.f((0, 0, 0))
    npt.assert_allclose(F[2, 2], (0.3 - 0.11) / 0.3, rtol=5e-5)


def test_FEBio_tied_elastic_contact_step(febio_cmd_xml):
    """E2E test of FEBio step-local tied elastic contact"""
    febio_cmd, xml_version = febio_cmd_xml
    # Test 1: Read
    mod = Path(__file__).with_suffix("").name
    pth_in = DIR_FIXTURES / f"{mod}.tied_elastic_contact_step.feb"
    model = wfl.load_model(pth_in)
    # Verify that contact exists in 2nd step
    assert len(model.steps[1].step.bc["contact"]) == 1
    # Test 2: Write
    pth_out = DIR_OUTPUT / f"{pth_in.stem}.{febio_cmd}.xml{xml_version}.feb"
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, xml_version)
    # Test 3: Solve - Can FEBio use the roundtripped file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd)
    # Test 4: Is the output as expected?
    solved = wfl.load_model(pth_out)
    # Test 4.0: Did the nodes of the rigid indenter move down?
    rigid_nodes = [i for face in model.named["face sets"].obj("indenter") for i in face]
    for idx in rigid_nodes:
        δz1 = solved.solution.values("displacement", idx)["displacement"][5][2]
        # Use loose tolerance b/c contact is imprecise
        npt.assert_allclose(δz1, -0.2, rtol=5e-5)
        δz2 = solved.solution.values("displacement", idx)["displacement"][-1][2] - δz1
        npt.assert_allclose(δz2, -0.11, rtol=5e-5)
    # Test 4.1: Did the top nodes of the deformable solid move down?
    top_nodes = [i for face in model.named["face sets"].obj("top") for i in face]
    for idx in top_nodes:
        δz = solved.solution.values("displacement", idx)["displacement"][-1][2]
        # Use loose tolerance b/c contact is imprecise
        npt.assert_allclose(δz, -0.11, rtol=5e-5)
    # Test 4.2: Was the correct strain produced?
    e_solid = [e for e in solved.mesh.elements if isinstance(e, wfl.element.Hex8)][0]
    F = e_solid.f((0, 0, 0))
    npt.assert_allclose(F[2, 2], (0.3 - 0.11) / 0.3, rtol=5e-5)


def test_FEBio_prescribe_rigid_body_displacement(febio_cmd_xml):
    """E2E test of prescribed rigid body displacement boundary condition"""
    febio_cmd, xml_version = febio_cmd_xml
    # Test 1: Read
    pth_in = (
        DIR_FIXTURES
        / f"{Path(__file__).with_suffix('').name}.prescribe_rigid_body_displacement.feb"
    )
    model = wfl.load_model(pth_in)
    # Verify that global rigid body fixed constraints were picked up
    for dof, var in [
        ("x2", "displacement"),
        ("α1", "rotation"),
        ("α2", "rotation"),
        ("α3", "rotation"),
    ]:
        assert len(model.fixed["body"][(dof, var)]) == 1
    # Verify that global variable body displacement constraints were picked up
    assert len(model.varying["body"]) == 1
    # Verify that step-specific rigid body prescribed constraints were picked up.
    assert len(model.steps[0].step.bc["body"]) == 1
    # Test 2: Write
    pth_out = DIR_THIS / "output" / f"{pth_in.stem}.{febio_cmd}.xml{xml_version}.feb"
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)
    # Test 3: Solve - Can FEBio use the roundtripped file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd)
    # Test 4: Is the output as expected?
    solved = wfl.load_model(pth_out)
    δz = solved.solution.values("displacement", 0)["displacement"][-1][2]
    npt.assert_almost_equal(δz, 0.43)
    δx = solved.solution.values("displacement", 0)["displacement"][-1][0]
    npt.assert_almost_equal(δx, 0.19)
    δy = solved.solution.values("displacement", 0)["displacement"][-1][1]
    npt.assert_almost_equal(δy, 0)


def test_FEBio_prescribe_node_pressure_Hex8(febio_cmd_xml):
    """E2E test of prescribed nodal pressure boundary condition"""
    febio_cmd, xml_version = febio_cmd_xml
    # Test 1: Read
    pth_in = DIR_FIXTURES / (
        f"{Path(__file__).with_suffix('').name}.prescribe_node_pressure.feb"
    )
    model = wfl.load_model(pth_in)
    #
    # Test 2: Write
    pth_out = (
        DIR_THIS
        / "output"
        / (
            f"{Path(__file__).with_suffix('').name}.prescribe_node_pressure"
            + f".{febio_cmd}.xml{xml_version}.feb"
        )
    )
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)
    # Test 3: Solve: Can FEBio use the roundtripped file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd)
    #
    # Test 4: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]
    # Test 4.1: Do we see the correct resultant fluid pressure?
    p_FEBio_1 = model.solution.value("fluid pressure", step=1, entity_id=0, region_id=1)
    npt.assert_almost_equal(p_FEBio_1, 50.0)
    p_FEBio_2 = model.solution.value("fluid pressure", step=2, entity_id=0, region_id=1)
    npt.assert_almost_equal(p_FEBio_2, 100.0)
    # Test 4.2: Do we see the correct resultant fluid flux?
    j_FEBio_1 = model.solution.value("fluid flux", step=1, entity_id=0, region_id=1)[0]
    j_FEBio_2 = model.solution.value("fluid flux", step=2, entity_id=0, region_id=1)[0]
    npt.assert_almost_equal(j_FEBio_1, 2.5)
    npt.assert_almost_equal(j_FEBio_2, 5.0)

# Python built-ins
import inspect
from pathlib import Path
from unittest import TestCase

# Published packages
import numpy as np
import numpy.testing as npt

# febtools' local modules
import febtools as feb
from febtools import Step
from febtools.load import prescribe_deformation
from febtools.control import auto_ticker
from febtools.febio import run_febio_checked
from febtools.math import vec_from_sph
from febtools.test.fixtures import (
    febio_cmd,
    gen_model_single_spiky_Hex8,
    RTOL_F,
    ATOL_F,
    RTOL_STRESS,
    ATOL_STRESS,
    DIR_FIXTURES,
    DIR_OUT,
)


def _fixture_FEBio_fiberDirectionLocal_Hex8_fiber():
    """Create fixture for FEBio_fiberDirectionLocal_Hex8_fiber test

    The choice of materials is inspired by the Elliott lab's
    intervertebral disc model.

    Run with:

        python -c "import febtools.test.test_material_axes; febtools.test.test_material_axes._fixture_FEBio_fiberDirectionLocal_Hex8_fiber()"

    """
    model = gen_model_single_spiky_Hex8()
    matrix = feb.material.HolmesMow({"E": 0.5, "v": 0, "beta": 3.4})
    fibers1 = feb.material.ExponentialFiber(
        {"alpha": 65, "beta": 2, "ksi": 0.296}, orientation=vec_from_sph(0, 60)
    )
    fibers2 = feb.material.ExponentialFiber(
        {"alpha": 65, "beta": 2, "ksi": 0.296}, orientation=vec_from_sph(0, 120)
    )
    fibers3 = feb.material.ExponentialFiber(
        {"alpha": 65, "beta": 2, "ksi": 0.296}, orientation=vec_from_sph(100, 90)
    )
    material = feb.material.SolidMixture([fibers1, fibers2, fibers3, matrix])
    for e in model.mesh.elements:
        e.material = material
    sequence = feb.Sequence(((0, 0), (1, 1)), extrap="extrapolate", interp="linear")
    model.add_step(Step(physics="solid", ticker=auto_ticker(sequence)))
    F = np.array([[1.14, 0.18, 0.11], [-0.20, 1.09, 0.17], [-0.11, 0.20, 1.12]])
    # ^ make all diagonal stretches large and positive to load fibers
    node_set = feb.NodeSet([i for i in range(len(model.mesh.nodes))])
    prescribe_deformation(model, node_set, F, sequence)
    # Write model to FEBio XML
    tree = feb.output.xml(model)
    pth = DIR_FIXTURES / (
        f"{Path(__file__).with_suffix('').name}." + "fiberDirectionLocal_Hex8_fiber.feb"
    )
    ## Add logfile output
    ##
    ## TODO: provide an interface for requesting logfile output
    e_Output = tree.find("Output")
    e_logfile = e_Output.makeelement("logfile")
    e_elementdata = e_logfile.makeelement(
        "element_data",
        file=pth.with_suffix("").name + "_-_element_data.txt",
        data="Fxx;Fyy;Fzz;Fxy;Fyz;Fxz;Fyx;Fzy;Fzx" + ";sx;sy;sz;sxy;sxz;syz",
    )
    e_logfile.append(e_elementdata)
    e_Output.append(e_logfile)
    with open(pth, "wb") as f:
        feb.output.write_xml(tree, f)
    run_febio_checked(pth)


def test_FEBio_SOHomFibAng_Hex8_ExpFiber(febio_cmd):
    """E2E test of 1D submaterial homogeneous orientation.

    Orientation given as <fiber type="angles"> in FEBio XML.

    """
    # Test 1: Read
    pth_in = DIR_FIXTURES / (
        f"{Path(__file__).with_suffix('').name}." + "SOHomFibAng_Hex8_ExpFiber.feb"
    )
    model = feb.load_model(pth_in)
    #
    # Test 2: Write
    pth_out = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        + "SOHomFibAng_Hex8_ExpFiber.{febio_cmd}.feb"
    )
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        feb.output.write_feb(model, f)
    # Test 3: Solve: Can FEBio use the roundtripped file?
    run_febio_checked(pth_out, cmd=febio_cmd)
    #
    # Test 4: Is the output as expected?
    model = feb.load_model(pth_out)
    e = model.mesh.elements[0]
    ##
    ## Test 4.1: Do we see the correct applied displacements?  A test
    ## failure here means that there is a defect in the code that reads
    ## or writes the model.  Or, less likely, an FEBio bug.
    ##
    ## Use 32-bit floats to match FEBio.
    FEBio_F_gpt_avg = np.array(
        [[1.14, 0.18, 0.11], [-0.2, 1.09, 0.17], [-0.11, 0.2, 1.12]], dtype=np.float32
    )
    F = np.mean([e.f(r).astype(np.float32) for r in e.gloc], axis=0).astype(np.float32)
    npt.assert_array_almost_equal_nulp(FEBio_F_gpt_avg, F, nulp=3)
    ##
    ## Test 4.2: Do we the correct output Cauchy stress?
    FEBio_cauchy_stress = np.array(
        [
            [854.5276, -222.80087, -515.7405],
            [-222.80087, 80.12013, 153.15533],
            [-515.7405, 153.15533, 401.15314],
        ]
    )
    # FEBio_cauchy_stress = model.solution.value('stress', -1, 0, 1)
    cauchy_stress_gpt = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    npt.assert_allclose(
        cauchy_stress_gpt, FEBio_cauchy_stress, rtol=RTOL_STRESS, atol=1e-3
    )


def test_FEBio_MOHomMatAxVec_Hex8_LinOrtho(febio_cmd):
    """E2E test of 3D material homogeneous orientation

    Orientation given as <mat_axis type="vector"> in FEBio XML.

    """
    # Test 1: Read
    pth_in = DIR_FIXTURES / (
        f"{Path(__file__).with_suffix('').name}." + "MOHomMatAxVec_Hex8_OrthoE.feb"
    )
    model = feb.load_model(pth_in)
    #
    # Test 2: Write
    pth_out = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        + f"MOHomMatAxVec_Hex8_OrthoE.{febio_cmd}.feb"
    )
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        feb.output.write_feb(model, f)
    # Test 3: Solve: Can FEBio use the roundtripped file?
    run_febio_checked(pth_out, cmd=febio_cmd)
    #
    # Test 4: Is the output as expected?
    model = feb.load_model(pth_out)
    e = model.mesh.elements[0]
    ##
    ## Test 4.1: Do we see the correct applied displacements?  A test
    ## failure here means that there is a defect in the code that reads
    ## or writes the model.  Or, less likely, an FEBio bug.
    FEBio_F_gpt_avg = np.array(
        [[1.14, 0.18, 0.11], [-0.2, 1.09, 0.17], [-0.11, 0.2, 1.12]]
    )
    F = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F, FEBio_F_gpt_avg, rtol=RTOL_F)
    ##
    ## Test 4.2: Do we the correct output Cauchy stress?
    FEBio_cauchy_stress = np.array(
        [
            [4.164129, -0.19348246, 0.19261515],
            [-0.19348246, 3.7601907, 4.5406365],
            [0.19261515, 4.5406365, 3.565557],
        ],
        dtype=np.float32,
    )
    # For unoriented case
    # FEBio_cauchy_stress = np.array([[ 3.2427673 , -0.13115434,  0.12526082],
    #                                 [-0.13115434,  3.580308  ,  3.7436085 ],
    #                                 [ 0.12526082,  3.7436085 ,  3.776064  ]])
    # FEBio_cauchy_stress = model.solution.value('stress', -1, 0, 1)
    cauchy_stress_gpt = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    npt.assert_allclose(
        cauchy_stress_gpt, FEBio_cauchy_stress, rtol=RTOL_STRESS, atol=ATOL_STRESS
    )


def test_FEBio_LOHetMatAxLoc_Hex8_OrthoE(febio_cmd):
    """E2E test of heterogeneous local basis.

    Heterogeneous local basis given as <mat_axis type="local"> in
    top-most material in FEBio XML.

    """
    # Test 1: Read
    pth_in = DIR_FIXTURES / (
        f"{Path(__file__).with_suffix('').name}." + "LOHetMatAxLoc_Hex8_OrthoE.feb"
    )
    model = feb.load_model(pth_in)
    #
    # Test 2: Write
    pth_out = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        + f"LOHetMatAxLoc_Hex8_OrthoE.{febio_cmd}.feb"
    )
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        feb.output.write_feb(model, f)
    # Test 3: Solve: Can FEBio use the roundtripped file?
    run_febio_checked(pth_out, cmd=febio_cmd)
    #
    # Test 4: Is the output as expected?
    model = feb.load_model(pth_out)
    ##
    ## Test 4.1: Do we see the correct applied displacements?  A test
    ## failure here means that there is a defect in the code that reads
    ## or writes the model.  Or, less likely, an FEBio bug.
    F_applied = np.array(
        [[1.14, 0.05, 0.03], [-0.02, 1.09, 0.02], [-0.01, -0.03, 1.12]]
    )
    for e in model.mesh.elements:
        F = np.mean([e.f(r) for r in e.gloc], axis=0)
        npt.assert_allclose(F, F_applied, rtol=RTOL_F, atol=ATOL_F)
    ##
    ## Test 4.2: Do we the correct output Cauchy stress?
    ##
    ## Test 4.2.1: Is the expected stress values in the xplt file?  A
    ## failure here implies a failure to read or write the heterogeneous
    ## local basis.
    σ_expected_E9 = np.array(
        [
            [2.8196144, 0.15565133, 0.15947175],
            [0.15565133, 1.7131327, -0.149578],
            [0.15947175, -0.149578, 2.4056263],
        ],
        dtype=np.float32,
    )
    σ_xplt_E9 = model.solution.value("stress", -1, 8, 1)
    npt.assert_allclose(σ_xplt_E9, σ_expected_E9, rtol=RTOL_STRESS)
    σ_expected_E12 = np.array(
        [
            [2.9279528, 0.39352927, 0.22572668],
            [0.39352927, 1.5805135, -0.05470883],
            [0.22572668, -0.05470883, 2.4882329],
        ],
        dtype=np.float32,
    )
    σ_xplt_E12 = model.solution.value("stress", -1, 11, 1)
    npt.assert_allclose(σ_xplt_E12, σ_expected_E12, rtol=RTOL_STRESS, atol=ATOL_STRESS)
    ## Test 4.2.2: Does febtools calculate the same stress as FEBio?
    for i in range(8, 12):
        e = model.mesh.elements[i]
        σ_FEBio = model.solution.value("stress", -1, i, 1)
        # e.material is just OrthotropicElastic; doesn't include local basis
        σ = np.mean([e.tstress(r) for r in e.gloc], axis=0)
        npt.assert_allclose(σ, σ_FEBio, rtol=RTOL_STRESS, atol=ATOL_STRESS)


def test_FEBio_LOHetMatAxLoc_SOHomFibAng_Hex8_PowLinFiber(febio_cmd):
    """E2E test of heterogeneous local basis + homogeneous 1D submaterial
    orientation.

        Heterogeneous local basis given as <mat_axis type="local"> in
        top-most material in FEBio XML.

        Homogeneous 1D submaterial orientation given as <fiber
        type="angles"> in FEBio XML.

    """
    # Test 1: Read
    pth_in = DIR_FIXTURES / (
        f"{Path(__file__).with_suffix('').name}."
        + "LOHetMatAxLoc_SOHomFibAng_Hex8_PowLinFiber.feb"
    )
    model = feb.load_model(pth_in)
    #
    # Test 2: Write
    pth_out = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        + f"LOHetMatAxLoc_SOHomFibAng_Hex8_PowLinFiber.{febio_cmd}.feb"
    )
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        feb.output.write_feb(model, f)
    # Test 3: Solve: Can FEBio use the roundtripped file?
    run_febio_checked(pth_out, cmd=febio_cmd)
    #
    # Test 4: Is the output as expected?
    model = feb.load_model(pth_out)
    ##
    ## Test 4.1: Do we see the correct applied displacements?  A test
    ## failure here means that there is a defect in the code that reads
    ## or writes the model.  Or, less likely, an FEBio bug.
    F_applied = np.array(
        [[1.14, 0.05, 0.03], [-0.02, 1.09, 0.02], [-0.01, -0.03, 1.12]]
    )
    for e in model.mesh.elements:
        F = np.mean([e.f(r) for r in e.gloc], axis=0)
        npt.assert_allclose(F, F_applied, rtol=RTOL_F, atol=ATOL_F)
    ##
    ## Test 4.2: Do we the correct output Cauchy stress?
    ##
    ## Test 4.2.1: Is the expected stress values in the xplt file?  A
    ## failure here implies a failure to read or write the heterogeneous
    ## local basis.
    σ_expected_E9 = np.array(
        [
            [0.16242483, 0.14094624, 0.00296378],
            [0.14094624, 0.13130957, 0.00079237],
            [0.00296378, 0.00079237, 0.2629625],
        ],
        dtype=np.float32,
    )
    σ_xplt_E9 = model.solution.value("stress", -1, 8, 1)
    npt.assert_allclose(σ_xplt_E9, σ_expected_E9, rtol=RTOL_STRESS, atol=ATOL_STRESS)
    σ_expected_E12 = np.array(
        [
            [0.00653787, 0.00997643, 0.00532855],
            [0.00997643, 0.21919638, -0.01209543],
            [0.00532855, -0.01209543, 0.225038],
        ],
        dtype=np.float32,
    )
    σ_xplt_E12 = model.solution.value("stress", -1, 11, 1)
    npt.assert_allclose(σ_xplt_E12, σ_expected_E12, rtol=RTOL_STRESS, atol=ATOL_STRESS)
    ## Test 4.2.2: Does febtools calculate the same stress as FEBio?
    for i in range(8, 12):
        e = model.mesh.elements[i]
        σ_FEBio = model.solution.value("stress", -1, i, 1)
        # e.material is just OrthotropicElastic; doesn't include local basis
        σ = np.mean([e.tstress(r) for r in e.gloc], axis=0)
        npt.assert_allclose(σ, σ_FEBio, rtol=RTOL_STRESS, atol=ATOL_STRESS)

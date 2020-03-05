# Python built-ins
import inspect
from pathlib import Path
from unittest import TestCase
# Published packages
import numpy as np
import numpy.testing as npt
# febtools' local modules
import febtools as feb
from febtools.conditions import prescribe_deformation
from febtools.control import auto_control_section
from febtools.febio import run_febio
from febtools.math import vec_from_sph
from febtools.test.fixtures import gen_model_single_spiky_Hex8


DIR_THIS = Path(__file__).parent
DIR_FIXTURES = Path(__file__).parent / "fixtures"
RTOL_STRESS = 5e-6

def _fixture_FEBio_fiberDirectionLocal_Hex8_fiber():
    """Create fixture for FEBio_fiberDirectionLocal_Hex8_fiber test

    The choice of materials is inspired by the Elliott lab's
    intervertebral disc model.

    Run with:

        python -c "import febtools.test.test_material_axes; febtools.test.test_material_axes._fixture_FEBio_fiberDirectionLocal_Hex8_fiber()"

    """
    model = gen_model_single_spiky_Hex8()
    matrix = feb.material.HolmesMow({"E": 0.5, "v": 0, "beta": 3.4})
    fibers1 = feb.material.ExponentialFiber({"alpha": 65,
                                             "beta": 2,
                                             "ksi": 0.296},
                                            orientation=vec_from_sph(0, 60))
    fibers2 = feb.material.ExponentialFiber({"alpha": 65,
                                             "beta": 2,
                                             "ksi": 0.296},
                                            orientation=vec_from_sph(0, 120))
    fibers3 = feb.material.ExponentialFiber({"alpha": 65,
                                             "beta": 2,
                                             "ksi": 0.296},
                                            orientation=vec_from_sph(100, 90))
    material = feb.material.SolidMixture([fibers1, fibers2, fibers3,
                                          matrix])
    for e in model.mesh.elements:
        e.material = material
    sequence = feb.Sequence(((0, 0), (1, 1)),
                            extend="extrapolate", typ="linear")
    model.add_step(control=auto_control_section(sequence, pts_per_segment=1))
    F = np.array([[1.14, 0.18, 0.11],
                  [-0.20, 1.09, 0.17],
                  [-0.11, 0.20, 1.12]])
    # ^ make all diagonal stretches large and positive to load fibers
    node_set = feb.NodeSet([i for i in range(len(model.mesh.nodes))])
    prescribe_deformation(model, node_set, F, sequence)
    # Write model to FEBio XML
    tree = feb.output.xml(model)
    pth = DIR_FIXTURES / \
        (f"{Path(__file__).with_suffix('').name}." +\
         "fiberDirectionLocal_Hex8_fiber.feb")
    ## Add logfile output
    ##
    ## TODO: provide an interface for requesting logfile output
    e_Output = tree.find("Output")
    e_logfile = e_Output.makeelement("logfile")
    e_elementdata = e_logfile.makeelement("element_data",
        file=pth.with_suffix("").name + "_-_element_data.txt",
        data="Fxx;Fyy;Fzz;Fxy;Fyz;Fxz;Fyx;Fzy;Fzx" + ";sx;sy;sz;sxy;sxz;syz")
    e_logfile.append(e_elementdata)
    e_Output.append(e_logfile)
    with open(pth, "wb") as f:
        feb.output.write_xml(tree, f)
    run_febio(pth)


def test_FEBio_SOHomFibAng_Hex8_fiber():
    """E2E test of submaterial orientation, <fiber type="angles">"""
    # Test 1: Read
    pth_in = DIR_FIXTURES / \
        (f"{Path(__file__).with_suffix('').name}." +\
         "SOHomFibAng_Hex8_fiber.feb")
    model = feb.load_model(pth_in)
    #
    # Test 2: Write
    pth_out = DIR_THIS / "output" / \
        (f"{Path(__file__).with_suffix('').name}." +\
         "SOHomFibAng_Hex8_fiber.feb")
    with open(pth_out, "wb") as f:
        feb.output.write_feb(model, f)
    # Test 3: Solve: Can FEBio use the roundtripped file?
    run_febio(pth_out)
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
    FEBio_F_gpt_avg = np.array([[ 1.14,  0.18, 0.11],
                                [-0.2 ,  1.09,  0.17],
                                [-0.11,  0.2 ,  1.12]], dtype=np.float32)
    F = np.mean([e.f(r).astype(np.float32) for r in e.gloc],
                axis=0).astype(np.float32)
    npt.assert_array_almost_equal_nulp(FEBio_F_gpt_avg, F, nulp=3)
    ##
    ## Test 4.2: Do we the correct output Cauchy stress?
    FEBio_cauchy_stress = np.array([[ 854.5276,  -222.80087, -515.7405 ],
                                    [-222.80087,   80.12013,  153.15533],
                                    [-515.7405,   153.15533,  401.15314]])
    # FEBio_cauchy_stress = model.solution.value('stress', -1, 0, 1)
    cauchy_stress_gpt = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    npt.assert_allclose(FEBio_cauchy_stress, cauchy_stress_gpt,
                        rtol=RTOL_STRESS)

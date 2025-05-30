from pathlib import Path
from unittest import TestCase

import pytest
from numpy.linalg import inv
import numpy.testing as npt

import waffleiron as wfl
from waffleiron import Model, Step
from waffleiron.control import auto_ticker
from waffleiron.febioxml import (
    UncoupledHGOFEBio,
    UncoupledMooneyRivlin,
    TransIsoMooneyRivlinFEBio,
)
from waffleiron.load import prescribe_deformation
from waffleiron.material import *
from waffleiron.mesh import rectangular_prism_hex8
from waffleiron.test.fixtures import (
    ATOL_F,
    RTOL_F,
    RTOL_STRESS,
    ATOL_STRESS,
    febio_4plus_cmd_xml,
    F_multiaxial,
    F_shear,
    F_rotations,
    F_monoaxial,
)
from waffleiron.input import FebReader, textdata_list
from waffleiron.test.fixtures import (
    DIR_FIXTURES,
    DIR_OUT,
    febio_cmd_xml,
    febio_3plus_cmd_xml,
    pytest_request,
)


def _create_model(mat, F):
    """Return model for given material and F tensor"""
    model = Model(rectangular_prism_hex8((1, 1, 1), ((0, 1), (0, 1), (0, 1))))
    model.mesh.elements[0].material = mat
    seq = wfl.Sequence(((0, 0), (1, 1)), interp="linear", extrap="constant")
    step = Step("solid", dynamics="static", ticker=auto_ticker(seq, 1))
    model.add_step(step)
    prescribe_deformation(model, step, np.arange(len(model.mesh.nodes)), F, seq)
    return model


def test_orthotropic_stiffness_compliance_equivalence():
    mat = OrthotropicLinearElastic(
        {
            "E1": 10,
            "E2": 9,
            "E3": 8,
            "G12": 5,
            "G23": 3,
            "G31": 4,
            "ν12": 0.3,
            "ν23": 0.2,
            "ν31": 0.4,
        }
    )
    C = orthotropic_elastic_stiffness_matrix_from_mat(mat)
    S = orthotropic_elastic_compliance_matrix_from_mat(mat)
    npt.assert_array_almost_equal(C, np.linalg.inv(S))
    npt.assert_array_almost_equal(S, np.linalg.inv(C))


class ExponentialFiberTest(TestCase):
    """Test exponential fiber material definition

    Since this material is unstable, it must be tested in a mixture.
    The test, therefore, is not truly independent.

    """

    def setUp(self):
        self.model = wfl.load_model(DIR_FIXTURES / "mixture_hm_exp.feb")
        self.soln = self.model.solution

    def w_test(self):
        # This is a very weak test; just a sanity check.
        expfib = ExponentialFiber(ξ=0.296, α=65, β=2)
        w = expfib.w(1.1)
        assert w > 0

    def test_tstress(self):
        """Check Cauchy stress against FEBio."""
        F = self.model.mesh.elements[0].f((0, 0, 0))
        t_try = self.model.mesh.elements[0].material.tstress(F)
        data = self.soln.step_data(-1)
        t_true = data[("stress", "domain")][0]
        npt.assert_allclose(t_try, t_true, rtol=1e-5, atol=1e-5)

    def test_sstress(self):
        """Check second Piola-Kirchoff stress via transform."""
        r = (0, 0, 0)
        elem = self.model.mesh.elements[0]
        f = elem.f(r)
        s_try = elem.material.sstress(f)
        t_try = (1.0 / np.linalg.det(f)) * np.dot(f, np.dot(s_try, f.T))
        t_true = self.soln.step_data(-1)[("stress", "domain")][0]
        npt.assert_allclose(t_try, t_true, rtol=1e-5, atol=1e-5)


class Unit_CauchyStress_PowerLinearFiber(TestCase):
    """Test piecewise power law – linear fibers.

    "True" values taken from PowerLinearFiber3D calculations.

    """

    def test_tstress_slack(self):
        """Check for lack of compressive resistance"""
        material = PowerLinearFiber(52, 2.5, 1.07)
        λ = 0.95
        expected = 0
        actual = material.stress(λ)
        assert expected == actual

    def test_tstress_origin(self):
        """Check for zero stress at zero strain"""
        material = PowerLinearFiber(52, 2.5, 1.07)
        λ = 0.95
        expected = 0
        actual = material.stress(λ)
        assert expected == actual

    def test_tstress_toe(self):
        material = PowerLinearFiber(52, 2.5, 1.07)
        λ = 1.05
        actual = material.stress(λ)
        expected = 1.21977973
        npt.assert_almost_equal(actual, expected, 5)

    def test_tstress_lin(self):
        material = PowerLinearFiber(52, 2.5, 1.07)
        λ = 1.12
        actual = material.stress(λ)
        expected = 4.21977234
        npt.assert_almost_equal(actual, expected, 5)


class Unit_CauchyStress_PowerLinearFiber3D(TestCase):
    """Test piecewise power law – linear fibers.

    Relevant fixture for "ground truth" values:
    test_material.PowerLinearFiberStress.feb

    """

    def test_tstress_slack(self):
        """Check for lack of compressive resistance"""
        material = PowerLinearFiber3D(52, 2.5, 1.07)
        F = np.array([[0.95, 0.0, 0.0], [0.0, 1.05, 0.0], [0.0, 0.0, 1.12]])
        # febio_stress = self.model.solution.value("stress", -1, 0, 1)
        expected = np.zeros((3, 3))
        actual = material.tstress(F)
        npt.assert_array_equal(expected, actual)

    def test_tstress_origin(self):
        """Check for zero stress at zero strain"""
        material = PowerLinearFiber3D(52, 2.5, 1.07)
        F = np.array([[1.0, 0.0, 0.0], [0.0, 1.05, 0.0], [0.0, 0.0, 1.12]])
        expected = np.zeros((3, 3))
        actual = material.tstress(F)
        npt.assert_array_equal(expected, actual)

    def test_tstress_toe(self):
        material = PowerLinearFiber3D(52, 2.5, 1.07, orientation=(0, 1, 0))
        F = np.array([[0.95, 0.0, 0.0], [0.0, 1.05, 0.0], [0.0, 0.0, 1.12]])
        actual = material.tstress(F)
        expected = np.array([[0, 0, 0], [0, 1.20373, 0], [0, 0, 0]])
        npt.assert_array_almost_equal(expected, actual, 5)

    def test_tstress_lin(self):
        material = PowerLinearFiber3D(52, 2.5, 1.07, orientation=(0, 0, 1))
        F = np.array([[0.95, 0.0, 0.0], [0.0, 1.05, 0.0], [0.0, 0.0, 1.12]])
        actual = material.tstress(F)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 4.73799]])
        npt.assert_array_almost_equal(expected, actual, 5)


class IsotropicElasticTest(TestCase):
    """Test isotropic elastic material definition"""

    def setUp(self):
        elemdata = textdata_list(
            DIR_FIXTURES / "isotropic_elastic_elem_data.txt", delim=","
        )
        febreader = FebReader(open(DIR_FIXTURES / "isotropic_elastic.feb"))
        mats, mat_labels = febreader.materials()
        self.mat = mats[0]
        Fxx = elemdata[-1]["Fxx"][0]
        Fyy = elemdata[-1]["Fyy"][0]
        Fzz = elemdata[-1]["Fzz"][0]
        Fxy = elemdata[-1]["Fxy"][0]
        Fxz = elemdata[-1]["Fxz"][0]
        Fyx = elemdata[-1]["Fyx"][0]
        Fyz = elemdata[-1]["Fyz"][0]
        Fzx = elemdata[-1]["Fzx"][0]
        Fzy = elemdata[-1]["Fzy"][0]
        F = np.array([[Fxx, Fxy, Fxz], [Fyx, Fyy, Fyz], [Fzx, Fzy, Fzz]])
        self.elemdata = elemdata
        self.F = F

    def test_props_conversion(self):
        """Test IsotropicElastic creation from E and ν."""
        youngmod = 1e6
        nu = 0.4
        y, mu = to_Lamé(youngmod, nu)
        matlprops = {"lambda": y, "mu": mu}
        mat1 = IsotropicElastic(matlprops)
        mat2 = self.mat
        w_try = mat1.w(self.F)
        w_true = mat2.w(self.F)
        npt.assert_approx_equal(w_try, w_true)

    def test_w_identity(self):
        F = np.eye(3)
        matprops = {"lambda": 1.0, "mu": 1.0}
        mat = IsotropicElastic(matprops)
        w_try = mat.w(F)
        w_true = 0.0
        npt.assert_approx_equal(w_try, w_true)

    def test_w(self):
        F = np.array([[1.1, 0.1, 0.0], [0.2, 0.9, 0.0], [-0.3, 0.0, 1.5]])
        matprops = {"lambda": 5.8e6, "mu": 3.8e6}
        mat = IsotropicElastic(matprops)
        W_try = mat.w(F)
        W_true = 3610887.5  # calculated by hand
        npt.assert_approx_equal(W_try, W_true)

    def test_sstress(self):
        tx = self.elemdata[-1]["sx"][0]
        ty = self.elemdata[-1]["sy"][0]
        tz = self.elemdata[-1]["sz"][0]
        txy = self.elemdata[-1]["sxy"][0]
        txz = self.elemdata[-1]["sxz"][0]
        tyz = self.elemdata[-1]["syz"][0]
        t_true = np.array([[tx, txy, txz], [txy, ty, tyz], [txz, tyz, tz]])
        Fdet = self.elemdata[-1]["J"][0]
        F = self.F
        s_true = Fdet * dot(inv(F), dot(t_true, inv(F.T)))
        s_try = self.mat.sstress(F)
        npt.assert_allclose(s_try, s_true, rtol=1e-3, atol=1.0)

    def test_tstress(self):
        """Compare calculated stress with that from FEBio's logfile"""
        # someday, the material properties will be read from the .feb
        # file
        tx = self.elemdata[-1]["sx"][0]
        ty = self.elemdata[-1]["sy"][0]
        tz = self.elemdata[-1]["sz"][0]
        txy = self.elemdata[-1]["sxy"][0]
        txz = self.elemdata[-1]["sxz"][0]
        tyz = self.elemdata[-1]["syz"][0]
        t_true = np.array([[tx, txy, txz], [txy, ty, tyz], [txz, tyz, tz]])
        F = self.F
        t_try = self.mat.tstress(F)
        npt.assert_allclose(t_try, t_true, rtol=1e-5)


class HolmesMowTest(TestCase):
    """Test Holmes Mow material definition"""

    def setUp(self):
        self.model = wfl.load_model(DIR_FIXTURES / "holmes_mow.feb")
        self.soln = self.model.solution

    def test_tstress(self):
        e = self.model.mesh.elements[0]
        F = e.f((0, 0, 0))
        t_try = e.material.tstress(F)
        t_true = self.soln.step_data(-1)[("stress", "domain")][0]
        npt.assert_allclose(t_try, t_true, rtol=1e-5)

    def test_sstress(self):
        """Check second Piola-Kirchoff stress via transform."""
        r = (0, 0, 0)
        elem = self.model.mesh.elements[0]
        f = elem.f(r)
        s_try = elem.material.sstress(f)
        t_try = (1.0 / np.linalg.det(f)) * np.dot(f, np.dot(s_try, f.T))
        t_true = self.soln.step_data(-1)[("stress", "domain")][0]
        npt.assert_allclose(t_try, t_true, rtol=1e-5)


class NeoHookeanTest(TestCase):
    """Test Holmes–Mow material definition"""

    def setUp(self):
        self.model = wfl.load_model(DIR_FIXTURES / "neo_hookean.feb")
        self.soln = self.model.solution

    def test_tstress(self):
        """Check Cauchy stress"""
        e = self.model.mesh.elements[0]
        F = e.f((0, 0, 0))
        t_try = e.material.tstress(F)
        t_true = self.soln.step_data(-1)[("stress", "domain")][0]
        npt.assert_allclose(t_try, t_true, rtol=1e-5)

    def test_sstress(self):
        """Check second Piola-Kirchoff stress via transform."""
        r = (0, 0, 0)
        elem = self.model.mesh.elements[0]
        f = elem.f(r)
        s_try = elem.material.sstress(f)
        t_try = (1.0 / np.linalg.det(f)) * np.dot(f, np.dot(s_try, f.T))
        t_true = self.soln.step_data(-1)[("stress", "domain")][0]
        npt.assert_allclose(t_try, t_true, rtol=1e-5)

    # Don't bother with 1st Piola-Kirchoff stress; it's implemented as a
    # transform, so the accepted value would just duplicate the
    # implementation.


def test_FEBio_Hex8_OrthoE(febio_cmd_xml):
    """E2E test of OrthotropicElastic material."""
    febio_cmd, xml_version = febio_cmd_xml
    # Test 1: Read
    pth_in = DIR_FIXTURES / (
        f"{Path(__file__).with_suffix('').name}." + "Hex8_OrthoE.feb"
    )
    model = wfl.load_model(pth_in)
    #
    # Test 2: Write
    pth_out = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        + f"Hex8_OrthoE.{febio_cmd}.xml{xml_version}.feb"
    )
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)
    # Test 3: Solve: Can FEBio use the roundtripped file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)
    #
    # Test 4: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]
    ##
    ## Test 4.1: Do we see the correct applied displacements?  A test
    ## failure here means that there is a defect in the code that reads
    ## or writes the model.  Or, less likely, an FEBio bug.
    F_applied = np.array([[1.11, 0.02, 0.05], [-0.10, 0.92, 0.07], [-0.06, 0.20, 1.07]])
    F = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F, F_applied, rtol=RTOL_F)
    ##
    ## Test 4.2: Do we the correct output Cauchy stress?
    FEBio_cauchy_stress = np.array(
        [
            [2.0102563, -0.87915385, -0.33581227],
            [-0.87915385, -0.09515007, 1.9258928],
            [-0.33581227, 1.9258928, 2.589372],
        ]
    )
    # FEBio_cauchy_stress = model.solution.value('stress', -1, 0, 1)
    cauchy_stress_gpt = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    npt.assert_allclose(cauchy_stress_gpt, FEBio_cauchy_stress, rtol=RTOL_STRESS)


class EllipsoidalPowerFiberBasic(TestCase):
    """Test EllipsoidalPowerFiber"""

    def setUp(self):
        self.m = EllipsoidalPowerFiber(ξ=(7, 5, 3), β=(3, 2.3, 2))

    def test_zero_strain_zero_stress(self):
        F = np.eye(3)
        σ = self.m.tstress(F)
        npt.assert_allclose(σ, np.zeros((3, 3)), atol=ATOL_STRESS)

    def test_compression_strain_zero_stress(self):
        F = np.array([[0.99, 0, 0], [0, 0.99, 0], [0, 0, 0.99]])
        σ = self.m.tstress(F)
        npt.assert_allclose(σ, np.zeros((3, 3)), atol=1e-7)

    def test_tension_strain_nonzero_stress(self):
        F = np.array([[1.01, 0, 0], [0, 1, 0], [0, 0, 1]])
        σ = self.m.tstress(F)
        assert np.all(np.diag(σ) > 0)
        npt.assert_allclose(σ[~np.eye(3, dtype=bool)], np.zeros(6), atol=ATOL_STRESS)
        assert σ[0, 0] == np.max(σ)


def test_FEBio_EllipsoidalPowerFiber(febio_cmd_xml):
    """E2E test of EllipsoidalPowerFiber material."""
    febio_cmd, xml_version = febio_cmd_xml
    F_applied = np.array([[1.08, 0, 0], [0, 0.97, 0], [0, 0, 0.88]])

    # Generate model
    mat = wfl.material.EllipsoidalPowerFiber(
        (1.55608279e02, 4.87349748e00, 2.00000000e01), (3, 2, 2.5)
    )
    model = _create_model(mat, F_applied)
    bn = f"{Path(__file__).with_suffix('').name}." + "EllPowFiber"

    # TODO: switch to roundtrip

    # Test 1: Write
    pth_out = DIR_OUT / (f"{bn}.{febio_cmd}.xml{xml_version}.feb")
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)

    # Test 2: Solve: Can FEBio use the file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)

    # Test 4: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]
    # Test 4.1: Do we see the correct applied displacements?  A test failure here
    # means that there is a defect in the code that reads or writes the model.  Or,
    # less likely, an FEBio bug.
    F_obs = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F_obs, F_applied, atol=ATOL_F)
    # Test 4.2: Do we see the correct stresses?
    # σ_wfl = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    σ_wfl = e.material.tstress(F_applied)
    σ_febio = model.solution.value("stress", step=1, entity_id=1, region_id=1)
    npt.assert_allclose(σ_wfl, σ_febio, atol=ATOL_STRESS)


@pytest.fixture(
    params=(
        np.diag([1.06, 1.08, 1.04]),
        np.diag([1.08, 0.92, 0.94]),
        np.diag([1.08, 1.06, 0.94]),
        np.diag([0.94, 0.96, 0.92]),
    ),
    ids=("T", "TC", "CT", "C"),
)
def F_cases_fibers(request):
    """F tensors for 3 tension–compression cases in Hou_Ateshian_2016"""
    F = request.param
    return F


def test_FEBio_FungOrthotropic(febio_cmd_xml, F_cases_fibers):
    """E2E test of FungOrthotropic material"""
    febio_cmd, xml_version = febio_cmd_xml
    F_applied = F_cases_fibers

    # Generate model
    mat = FungOrthotropicElastic(
        E1=10, E2=9, E3=8, G12=5, G23=3, G31=4, ν12=0.3, ν23=0.2, ν31=0.4, c=3, K=0.3
    )
    model = _create_model(mat, F_applied)
    bn = f"{Path(__file__).with_suffix('').name}." + "FungOrtho"

    # TODO: switch to roundtrip

    # Test 1: Write
    pth_out = DIR_OUT / (f"{bn}.{febio_cmd}.xml{xml_version}.feb")
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)

    # Test 2: Solve: Can FEBio use the file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)

    # Test 4: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]
    # Test 4.1: Do we see the correct applied displacements?  A test failure here
    # means that there is a defect in the code that reads or writes the model.  Or,
    # less likely, an FEBio bug.
    F_obs = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F_obs, F_applied, atol=ATOL_F)
    # Test 4.2: Do we see the correct stresses?
    σ_wfl = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    σ_febio = model.solution.value("stress", step=1, entity_id=1, region_id=1)
    # rtol = 0 due to underflow with zero stress
    npt.assert_allclose(σ_wfl, σ_febio, atol=ATOL_STRESS, rtol=0)


def test_FEBio_EllipsoidalDistribution(febio_cmd_xml, F_cases_fibers):
    """E2E test of EllipsoidalPowerFiber material"""
    febio_cmd, xml_version = febio_cmd_xml
    F_applied = F_cases_fibers

    # Generate model
    matrix = NeoHookean(1, 0)
    fibers = EllipsoidalDistribution(
        [1, 0.1, 0.1], PowerLinearFiber(E=100, β=2.5, λ0=1.01)
    )
    mat = SolidMixture([matrix, fibers])
    model = _create_model(mat, F_applied)
    bn = f"{Path(__file__).with_suffix('').name}." + "EllDist"

    # TODO: switch to roundtrip

    # Test 1: Write
    pth_out = DIR_OUT / (f"{bn}.{febio_cmd}.xml{xml_version}.feb")
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)

    # Test 2: Solve: Can FEBio use the file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)

    # Test 4: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]
    # Test 4.1: Do we see the correct applied displacements?  A test failure here
    # means that there is a defect in the code that reads or writes the model.  Or,
    # less likely, an FEBio bug.
    F_obs = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F_obs, F_applied, atol=ATOL_F)
    # Test 4.2: Do we see the correct stresses?
    σ_wfl = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    σ_febio = model.solution.value("stress", step=1, entity_id=1, region_id=1)
    # rtol = 0 due to underflow with zero stress
    npt.assert_allclose(σ_wfl, σ_febio, atol=5e-3, rtol=0)


def test_FEBio_UncoupledMooneyRivlin(febio_3plus_cmd_xml):
    """E2E test of UncoupledHGOFEBio material"""
    febio_cmd, xml_version = febio_3plus_cmd_xml
    F = np.diag([1.07, 1.2, 0.97])

    # Generate model
    mat = UncoupledMooneyRivlin(
        c1=5,  # MPa
        c2=3,  # MPa
        bulk=VolumetricLinear(1.5),
    )
    model = _create_model(mat, F)
    bn = f"{Path(__file__).with_suffix('').name}." + "DeviatoricMooneyRivlin"

    # TODO: switch to roundtrip

    # Test 1: Write
    pth_out = DIR_OUT / (f"{bn}.{febio_cmd}.xml{xml_version}.feb")
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)

    # Test 2: Solve: Can FEBio use the file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)

    # Test 3: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]

    # Test 4.1: Do we see the correct applied displacements?  A test failure here
    # means that there is a defect in the code that reads or writes the model.  Or,
    # less likely, an FEBio bug.
    F_obs = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F_obs, F, atol=ATOL_F)
    # Test 4.2: Do we see the correct stresses?
    σ_wfl = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    σ_febio = model.solution.value("stress", step=1, entity_id=1, region_id=1)
    npt.assert_allclose(σ_wfl, σ_febio, atol=ATOL_STRESS, rtol=0)


@pytest.mark.parametrize(
    "F",
    (
        np.diag([1.02, 1, 1]),
        np.diag([1.100, 1, 1]),
        np.array([[0.75, 0.03, 0.7], [0.025, 1.02, 0.06], [-0.7, 0.07, 0.7]]),
    ),
)
def test_FEBio_UncoupledTransIsoMooneyRivlin(pytest_request, febio_3plus_cmd_xml, F):
    """E2E test of FEBio "trans iso Mooney-Rivlin" material"""
    febio_cmd, xml_version = febio_3plus_cmd_xml
    # Generate model
    mat = TransIsoMooneyRivlinFEBio(
        c1=4.61,  # MPa
        c2=2.91,  # MPa
        c3=0.1197,
        c4=150,
        λ1=1.04,
        c5=400,
        bulk=VolumetricLogInverse(1.52e-19),
    )
    model = _create_model(mat, F)
    nm = f"{Path(__file__).with_suffix('').name}." + pytest_request.node.name

    # TODO: switch to roundtrip

    # Test 1: Write
    pth_out = DIR_OUT / (f"{nm}.feb")
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)

    # Test 2: Solve: Can FEBio use the file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)

    # Test 3: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]

    # Test 4.1: Do we see the correct applied displacements?  A test failure here
    # means that there is a defect in the code that reads or writes the model.  Or,
    # less likely, an FEBio bug.
    F_obs = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F_obs, F, atol=ATOL_F)
    # Test 4.2: Do we see the correct stresses?
    σ_wfl = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    σ_febio = model.solution.value("stress", step=1, entity_id=1, region_id=1)
    npt.assert_allclose(σ_wfl, σ_febio, atol=5e-6, rtol=0)


def test_FEBio_UncoupledHGO(febio_3plus_cmd_xml, F_cases_fibers):
    """E2E test of UncoupledHGOFEBio material"""
    febio_cmd, xml_version = febio_3plus_cmd_xml
    F_applied = F_cases_fibers

    # Generate model
    mat = UncoupledHGOFEBio(
        c=0.5,  # MPa
        k1=140,  # MPa,
        k2=2,
        γ=15,  # °
        κ=0.2,
        K=0.1,  # MPa
    )
    model = _create_model(mat, F_applied)
    bn = f"{Path(__file__).with_suffix('').name}." + "UncoupledHGO"

    # TODO: switch to roundtrip

    # Test 1: Write
    pth_out = DIR_OUT / (f"{bn}.{febio_cmd}.xml{xml_version}.feb")
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)

    # Test 2: Solve: Can FEBio use the file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)

    # Test 3: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]

    # Test 4.1: Do we see the correct applied displacements?  A test failure here
    # means that there is a defect in the code that reads or writes the model.  Or,
    # less likely, an FEBio bug.
    F_obs = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F_obs, F_applied, atol=ATOL_F)
    # Test 4.2: Do we see the correct stresses?
    σ_wfl = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    σ_febio = model.solution.value("stress", step=1, entity_id=1, region_id=1)
    npt.assert_allclose(σ_wfl, σ_febio, atol=5e-3)


def test_FEBio_NeoHookeanFiber(febio_3plus_cmd_xml, F_cases_fibers):
    """E2E test of LogarithmicFiber material"""
    febio_cmd, xml_version = febio_3plus_cmd_xml
    F_applied = F_cases_fibers

    mat = OrientedMaterial(NeoHookeanFiber(E=7), Q=np.array([0.5, 0.5, 2**0.5 / 2]))
    model = _create_model(mat, F_applied)
    bn = f"{Path(__file__).with_suffix('').name}." + "LogarithmicFiber"

    # TODO: switch to roundtrip

    # Test 1: Write
    pth_out = DIR_OUT / (f"{bn}.{febio_cmd}.xml{xml_version}.feb")
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)

    # Test 2: Solve: Can FEBio use the file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)

    # Test 4: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]
    # Test 4.1: Do we see the correct applied displacements?  A test failure here
    # means that there is a defect in the code that reads or writes the model.  Or,
    # less likely, an FEBio bug.
    F_obs = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F_obs, F_applied, atol=ATOL_F)
    # Test 4.2: Do we see the correct stresses?
    σ_wfl = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    σ_febio = model.solution.value("stress", step=1, entity_id=1, region_id=1)
    npt.assert_allclose(σ_wfl, σ_febio, atol=ATOL_STRESS)


def test_FEBio_LogInv2Fiber(febio_4plus_cmd_xml, F_cases_fibers):
    """E2E test of LogarithmicFiber material"""
    febio_cmd, xml_version = febio_4plus_cmd_xml
    F_applied = F_cases_fibers

    mat = OrientedMaterial(
        NaturalNeoHookeanFiber(E=0.33, λ0=1.032), Q=np.array([0.5, 0.5, 2**0.5 / 2])
    )
    model = _create_model(mat, F_applied)
    bn = f"{Path(__file__).with_suffix('').name}." + "LogarithmicFiber"

    # TODO: switch to roundtrip

    # Test 1: Write
    pth_out = DIR_OUT / (f"{bn}.{febio_cmd}.xml{xml_version}.feb")
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)

    # Test 2: Solve: Can FEBio use the file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)

    # Test 4: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]
    # Test 4.1: Do we see the correct applied displacements?  A test failure here
    # means that there is a defect in the code that reads or writes the model.  Or,
    # less likely, an FEBio bug.
    F_obs = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F_obs, F_applied, atol=ATOL_F)
    # Test 4.2: Do we see the correct stresses?
    σ_wfl = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    σ_febio = model.solution.value("stress", step=1, entity_id=1, region_id=1)
    npt.assert_allclose(σ_wfl, σ_febio, atol=ATOL_STRESS)


@pytest.mark.parametrize("F", (np.diag([1.03, 1, 1]), np.diag([1.10, 1, 1])))
def test_FEBio_ExpεAndLinεDEFiber(febio_cmd_xml, F):
    """E2E test of ExpεAndLinεDEFiber material"""
    febio_cmd, xml_version = febio_cmd_xml

    mat = OrientedMaterial(
        ExpAndLinearDCFiber(ξ=0.21, α=9, λ1=1.05, E=19),
        Q=np.array([1, 0, 0]),
    )
    model = _create_model(mat, F)
    bn = f"{Path(__file__).with_suffix('').name}." + "ExpεAndLinεDEFiber"

    # TODO: switch to roundtrip

    # Test 1: Write
    pth_out = DIR_OUT / (f"{bn}.{febio_cmd}.xml{xml_version}.feb")
    if not pth_out.parent.exists():
        pth_out.parent.mkdir()
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)

    # Test 2: Solve: Can FEBio use the file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)

    # Test 4: Is the output as expected?
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]
    # Test 4.1: Do we see the correct applied displacements?  A test failure here
    # means that there is a defect in the code that reads or writes the model.  Or,
    # less likely, an FEBio bug.
    F_obs = np.mean([e.f(r) for r in e.gloc], axis=0)
    npt.assert_allclose(F_obs, F, atol=ATOL_F)
    # Test 4.2: Do we see the correct stresses?
    σ_wfl = np.mean([e.material.tstress(e.f(r)) for r in e.gloc], axis=0)
    σ_febio = model.solution.value("stress", step=1, entity_id=1, region_id=1)
    npt.assert_allclose(σ_wfl, σ_febio, atol=ATOL_STRESS)


##############################################
# Transversely Isotropic General Exponential #
##############################################

materials_with_material_elasticity = {
    "IsotropicElastic": lambda: IsotropicElastic({"E": 5, "v": 0.33}),
    "TransIsoExponential": lambda: TransIsoExponential(
        0.08, 10, 23.3, 2.5, 47, 335, -23
    ),
}


@pytest.mark.parametrize(
    "material_factory",
    list(materials_with_material_elasticity.values()),
    ids=list(materials_with_material_elasticity.keys()),
)
@pytest.mark.parametrize(
    "F", F_monoaxial + F_shear + tuple(R @ F_multiaxial[0] for R in F_rotations)
)
def test_elasticity_properties(material_factory, F):
    """Check elasticity tensor properties"""
    material = material_factory()
    # material itself checks for positive definiteness in natural state
    C = material.material_elasticity(F)
    # np.set_printoptions(linewidth=120, precision=2)
    # print(f"\n{to_voigt_matrix(C)}")
    tens4_is_major_symmetric(C, as_assert=True)
    tens4_is_left_minor_symmetric(C, as_assert=True)
    tens4_is_right_minor_symmetric(C, as_assert=True)
    # these F and material parameters are chosen so material is stable; if the
    # compliance matrix is not positive definite, something probably broke somewhere
    assert is_positive_definite(C)

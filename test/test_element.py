import os, random
from pathlib import Path
from shutil import copyfile
import unittest

import numpy as np
import numpy.testing as npt
from lxml import etree

import waffleiron as wfl
from waffleiron.febio import run_febio_checked

from waffleiron.test.fixtures import (
    RTOL_STRESS,
    ATOL_STRESS,
    DIR_OUT,
    febio_cmd,
)


DIR_THIS = Path(__file__).parent
DIR_FIXTURES = Path(__file__).parent / "fixtures"


def f_tensor_logfile(elemdata, step, eid):
    """Return F tensor from logfile data

    Read the logfile first with textdata_list.  This function returns
    the F tensor as a 3×3 array for the requested timepoint and element
    ID.

    """
    Fxx = elemdata[step]["Fxx"][eid]
    Fyy = elemdata[step]["Fyy"][eid]
    Fzz = elemdata[step]["Fzz"][eid]
    Fxy = elemdata[step]["Fxy"][eid]
    Fxz = elemdata[step]["Fxz"][eid]
    Fyx = elemdata[step]["Fyx"][eid]
    Fyz = elemdata[step]["Fyz"][eid]
    Fzx = elemdata[step]["Fzx"][eid]
    Fzy = elemdata[step]["Fzy"][eid]
    F = np.array([[Fxx, Fxy, Fxz], [Fyx, Fyy, Fyz], [Fzx, Fzy, Fzz]])
    return F


class Hex8ElementTest(unittest.TestCase):
    def setUp(self):
        nodes = [
            (-2, -1.5, -3),
            (2, -1.5, -3),
            (2, 1.0, -3),
            (-2, 1.0, -3),
            (-2, -1.5, 1.2),
            (2, -1.5, 1.2),
            (2, 1.0, 1.2),
            (-2, 1.0, 1.2),
        ]
        m = wfl.material.IsotropicElastic({"E": 1e8, "v": 0.3})
        self.element = wfl.element.Hex8(nodes, material=m)

        # apply displacement
        self.ftensor = np.array(
            [[1.1, 0.2, 0.1], [0.2, 0.9, -0.14], [0.1, -0.14, 1.02]]
        )
        d = np.array([np.dot(self.ftensor - np.eye(3), node) for node in nodes])
        self.element.properties["displacement"] = d

        # Calculate nodal stress tensors.  These will actually all be
        # the same here, since the applied displacements result in a
        # uniform f tensor.
        nodal_f = np.array([self.element.f(r) for r in self.element.vloc])
        nodal_s = np.array([self.element.material.sstress(f) for f in nodal_f])
        self.element.properties["S"] = nodal_s

        self.w = 4.0
        self.l = 2.5
        self.h = 4.2

        c = np.array([10.0, 2.0, -1.4, 3.1, -1.0, 2.5, 0.7, 0.5])

        def fn(p, c=c):
            return (
                c[0]
                + c[1] * p[0]
                + c[2] * p[1]
                + c[3] * p[2]
                + c[4] * p[0] * p[1]
                + c[5] * p[0] * p[2]
                + c[6] * p[1] * p[2]
                + c[7] * p[0] * p[1] * p[2]
            )

        def dfn(p, c=c):
            a = np.array(
                [
                    c[1] + c[4] * p[1] + c[5] * p[2] + c[7] * p[1] * p[2],
                    c[2] + c[4] * p[0] + c[6] * p[2] + c[7] * p[0] * p[2],
                    c[3] + c[5] * p[0] + c[6] * p[1] + c[7] * p[0] * p[1],
                ]
            )
            return a

        def ddfn(p, c=c):
            a = np.array(
                [
                    [0.0, c[4] + c[7] * p[2], c[5] + c[7] * p[1]],
                    [c[4] + c[7] * p[2], 0.0, c[6] + c[7] * p[0]],
                    [c[5] + c[7] * p[1], c[6] + c[7] * p[0], 0.0],
                ]
            )
            return a

        self.fn = fn
        self.dfn = dfn
        self.ddfn = ddfn

        v = np.array([fn(pt) for pt in self.element.x()])
        self.element.properties["scalar_test"] = v

    def test_j(self):
        desired = np.array([[self.w / 2, 0, 0], [0, self.l / 2, 0], [0, 0, self.h / 2]])
        # at center
        actual = self.element.j((0, 0, 0), config="reference")
        npt.assert_allclose(actual, desired)
        # at gauss points
        for pt in self.element.gloc:
            actual = self.element.j(pt, config="reference")
            npt.assert_allclose(actual, desired, atol=np.spacing(1))

    def test_shape_fn(self):
        """Test calculation of node positions from nat coords."""
        nodes = self.element.x()
        for node, r in zip(nodes, self.element.vloc):
            desired = np.dot(self.element.x().T, self.element.N(*r))
            actual = node
            npt.assert_allclose(actual, desired)

    def test_dinterp_scalar(self):
        """Test first derivative against linear gradients."""
        for r in self.element.gloc:
            pt = self.element.interp(r, prop="position")
            desired = self.dfn(pt)
            actual = self.element.dinterp(r, prop="scalar_test")
            npt.assert_allclose(actual, desired)

    def test_ddinterp_scalar(self):
        """Test the second derivative against linear gradients.

        The second derivative of a linear gradient should equal zero.

        """
        for r in self.element.gloc:
            pt = self.element.interp(r, prop="position")
            desired = self.ddfn(pt)
            actual = self.element.ddinterp(r, prop="scalar_test")
            npt.assert_allclose(actual, desired, atol=10 * np.finfo(float).eps)

    def test_dinterp_vector(self):
        dudx = self.element.dinterp((0, 0, 0), prop="displacement")
        f = dudx + np.eye(3)
        npt.assert_allclose(f, self.ftensor)

    def test_ddinterp_vector(self):
        actual = self.element.ddinterp((0, 0, 0), prop="displacement")
        assert actual.shape == (3, 3, 3)
        # TODO: check value

    def test_dinterp_tensor(self):
        """Check 1st derivative of stress tensor."""
        random.seed(0)
        # Construct node-valued tensors with known gradient
        desired = np.zeros((3, 3, 3))
        nodal_tensors = np.zeros((8, 3, 3))
        r_pt = (0, 0, 0)
        x_pt = self.element.interp(r_pt, prop="position")
        for i in range(3):
            for j in range(3):
                c = [random.randrange(-100, 100) / 10.0 for a in range(8)]
                desired[i, j, :] = self.dfn(x_pt, c=c)
                for k, x_p in enumerate(self.element.x()):
                    nodal_tensors[k, i, j] = self.fn(x_p, c=c)
        self.element.properties["tensor"] = nodal_tensors

        actual = self.element.dinterp(r_pt, prop="tensor")
        # check shape
        assert actual.shape == (3, 3, 3)
        # check value
        npt.assert_almost_equal(actual, desired)

    def test_ddinterp_tensor(self):
        """Test 2nd derivative of tensor (Hex8)."""
        random.seed(0)
        r_pt = (0, 0, 0)
        x_pt = self.element.interp(r_pt, prop="position")

        # Construct node-valued tensors with known spatial second
        # derivative
        desired = np.zeros((3, 3, 3, 3))
        nodal_tensors = np.zeros((8, 3, 3))
        for i in range(3):
            for j in range(3):
                c = [random.randrange(-100, 100) / 10.0 for a in range(8)]
                desired[i, j, ...] = self.ddfn(x_pt, c=c)
                for k, x_p in enumerate(self.element.x()):
                    nodal_tensors[k, i, j] = self.fn(x_p, c=c)
        self.element.properties["tensor"] = nodal_tensors

        actual = self.element.ddinterp(r_pt, prop="tensor")
        # check shape
        assert actual.shape == (3, 3, 3, 3)
        # check value
        npt.assert_almost_equal(actual, desired)

    def test_integration_volume(self):
        truth = self.w * self.l * self.h
        computed = self.element.integrate(lambda e, r: 1.0)
        npt.assert_approx_equal(computed, truth)


class Quad4ElementTiltedTest(unittest.TestCase):
    """Test Quad4 element functions in R3.

    The element is tilted to ensure that embedding in R3 works
    correctly.

    """

    def setUp(self):
        nodes = [(-2, -1.5, 0), (2, -1.5, 1.0), (2, 1.0, 3.5), (-2, 1.0, 2.5)]
        self.element = wfl.element.Quad4(nodes)

    def test_j(self):
        desired = np.array([[2.0, 0], [0, (1.0 - -1.5) / 2.0], [0.5, 2.5 / 2.0]])
        # at center
        actual = self.element.j((0, 0, 0), config="reference")
        npt.assert_allclose(actual, desired)
        # at gauss points
        for pt in self.element.gloc:
            actual = self.element.j(pt, config="reference")
            npt.assert_allclose(actual, desired, atol=np.spacing(1))


class Quad4ElementTest(unittest.TestCase):
    """Test Quad4 element functions.

    The element is in the x-y plane to make creating synthetic
    derivatives easier.

    """

    def setUp(self):
        nodes = [(-2, -1.5, 2.0), (2, -1.5, 2.0), (2, 1.0, 2.0), (-2, 1.0, 2.0)]
        m = wfl.material.IsotropicElastic({"E": 1e8, "v": 0.3})
        self.element = wfl.element.Quad4(nodes, material=m)

        self.w = 4.0
        self.l = 2.5

        random.seed(10)
        c = [random.randrange(-100, 100) / 10.0 for i in range(5)]

        def fn(p, c=c):
            return c[0] + c[1] * p[0] + c[2] * p[1] + c[3] * p[2] + c[4] * p[0] * p[1]

        def dfn(p, c=c):
            a = np.array([c[1] + c[4] * p[1], c[2] + c[4] * p[0], 0.0])
            return a

        def ddfn(p, c=c):
            a = np.array([[0.0, c[4], 0.0], [c[4], 0.0, 0.0], [0.0, 0.0, 0.0]])
            return a

        self.fn = fn
        self.dfn = dfn
        self.ddfn = ddfn

        # Assign scalar values
        v = np.array([fn(pt) for pt in self.element.x()])
        self.element.properties["scalar_test"] = v

        # apply displacement
        self.ftensor = np.array([[1.1, -0.2, 0.0], [-0.2, 0.9, 0.0], [0.0, 0.0, 1.0]])
        d = np.array([np.dot(self.ftensor - np.eye(3), node) for node in nodes])
        self.element.properties["displacement"] = d

    def test_j(self):
        desired = np.array([[2.0, 0], [0, (1.0 - -1.5) / 2.0], [0.0, 0.0]])
        # at center
        actual = self.element.j((0, 0, 0), config="reference")
        npt.assert_allclose(actual, desired)
        # at gauss points
        for pt in self.element.gloc:
            actual = self.element.j(pt, config="reference")
            npt.assert_allclose(actual, desired, atol=np.spacing(1))

    def test_shape_fn(self):
        """Test calculation of node positions from nat coords."""
        nodes = self.element.x()
        for node, r in zip(nodes, self.element.vloc):
            desired = np.dot(self.element.x().T, self.element.N(*r))
            actual = node
            npt.assert_allclose(actual, desired)

    def test_dinterp_scalar(self):
        """Test first derivative against linear gradients."""
        for r in self.element.gloc:
            pt = self.element.interp(r, prop="position")
            desired = self.dfn(pt)
            actual = self.element.dinterp(r, prop="scalar_test")
            npt.assert_allclose(actual, desired, atol=10 * np.finfo(float).eps)

    def test_ddinterp_scalar(self):
        """Test the second derivative against linear gradients.

        The second derivative of a linear gradient should equal zero.

        """
        for r in self.element.gloc:
            pt = self.element.interp(r, prop="position")
            desired = self.ddfn(pt)
            actual = self.element.ddinterp(r, prop="scalar_test")
            npt.assert_allclose(actual, desired, atol=10 * np.finfo(float).eps)

    def test_dinterp_vector(self):
        dudx = self.element.dinterp((0, 0, 0), prop="displacement")
        f = dudx + np.eye(3)
        npt.assert_allclose(f, self.ftensor)

    def test_ddinterp_vector(self):
        actual = self.element.ddinterp((0, 0, 0), prop="displacement")
        assert actual.shape == (3, 3, 3)
        # TODO: check value

    def test_dinterp_tensor(self):
        """Test 1st derivative of stress tensor (Quad4)."""
        random.seed(0)
        # Construct node-valued tensors with known gradient
        desired = np.zeros((3, 3, 3))
        nodal_tensors = np.zeros((self.element.n, 3, 3))
        r_pt = (0, 0, 0)
        x_pt = self.element.interp(r_pt, prop="position")
        for i in range(3):
            for j in range(3):
                c = [random.randrange(-100, 100) / 10.0 for a in range(5)]
                desired[i, j, :] = self.dfn(x_pt, c=c)
                for k, x_p in enumerate(self.element.x()):
                    nodal_tensors[k, i, j] = self.fn(x_p, c=c)
        self.element.properties["tensor"] = nodal_tensors

        actual = self.element.dinterp(r_pt, prop="tensor")
        # check shape
        assert actual.shape == (3, 3, 3)
        # check value
        npt.assert_almost_equal(actual, desired)

    def test_ddinterp_tensor(self):
        """Check 2nd derivative of stress tensor."""
        random.seed(0)
        r_pt = (0, 0, 0)
        x_pt = self.element.interp(r_pt, prop="position")

        # Construct node-valued tensors with known spatial second
        # derivative
        desired = np.zeros((3, 3, 3, 3))
        nodal_tensors = np.zeros((self.element.n, 3, 3))
        for i in range(3):
            for j in range(3):
                c = [random.randrange(-100, 100) / 10.0 for a in range(5)]
                desired[i, j, ...] = self.ddfn(x_pt, c=c)
                for k, x_p in enumerate(self.element.x()):
                    nodal_tensors[k, i, j] = self.fn(x_p, c=c)
        self.element.properties["tensor"] = nodal_tensors

        actual = self.element.ddinterp(r_pt, prop="tensor")
        # check shape
        assert actual.shape == (3, 3, 3, 3)
        # check value
        npt.assert_almost_equal(actual, desired)

    def test_integration_area(self):
        truth = self.w * self.l
        computed = self.element.integrate(lambda e, r: 1.0)
        npt.assert_approx_equal(computed, truth)


def test_FEBio_F_Hex8():
    """Test calculation of the F tensor.

    F tensors are computed for each element based on the displacement
    data read from the XPLT file and compared to the F tensor values
    recorded in the text log.

    """
    # Setup
    bname = "bar_explicit_rb_grip_twist_stretch"
    pth_in = DIR_FIXTURES / f"{bname}.feb"
    pth_out = DIR_OUT / pth_in.name
    model = wfl.load_model(pth_in)
    with open(pth_out, "wb") as f:
        wfl.output.write_feb(model, f, version="3.0")
    xml = wfl.input.read_febio_xml(pth_out)
    e_logfile = etree.SubElement(xml.find("Output"), "logfile")
    e_edata = etree.SubElement(e_logfile, "element_data")
    e_edata.attrib["file"] = f"{bname}_-_elem_data.txt"
    e_edata.attrib["delim"] = ", "
    e_edata.attrib["data"] = (
        "x;y;z;sx;sy;sz;sxy;syz;sxz;s1;s2;s3;Ex;Ey;Ez;Exy;Eyz;Exz;E1;E2;E3;Fxx;Fyy;Fzz;Fxy;Fxz;Fyx;Fyz;Fzx;Fzy;J"
    )
    e_rbdata = etree.SubElement(e_logfile, "rigid_body_data")
    e_rbdata.attrib["file"] = f"{bname}_-_body_data.txt"
    e_rbdata.attrib["delim"] = ", "
    e_rbdata.attrib["data"] = "x;y;z;Fx;Fy;Fz;Mx;My;Mz"
    with open(pth_out, "wb") as f:
        wfl.output.write_xml(xml, f)
    wfl.febio.run_febio_checked(pth_out, threads=1)
    elemdata = wfl.input.textdata_list(DIR_OUT / f"{bname}_-_elem_data.txt", delim=",")
    # Check F tensor values
    model = wfl.load_model(pth_out)
    istep = -1
    for eid in range(len(model.mesh.elements) - 1):
        # ^ Use len - 1 so rigid body (last element) is not checked
        F_expected = f_tensor_logfile(elemdata, istep, eid)
        F = model.mesh.elements[eid].f((0, 0, 0))
        npt.assert_almost_equal(F, F_expected, decimal=5)


def test_FEBio_intraElementHetF_Hex8(febio_cmd):
    """Test handling of intra-element F-tensor heterogeneity.

    FEBio evaluates element data values at the Gauss points and reports the average
    over the Gauss points in its logfile and plotfile output.  This Gauss point
    averaging is not readily apparent in the F tensor output, but is very obvious in
    the strain and stress tensors.

    """
    pth_in = DIR_FIXTURES / "test_element.intraElementHetF_Hex8.feb"
    pth_out = DIR_OUT / f"test_element.intraElementHetF_Hex8.{febio_cmd}.feb"
    copyfile(pth_in, pth_out)
    run_febio_checked(pth_out, cmd=febio_cmd, threads=1)
    model = wfl.load_model(pth_out)
    e = model.mesh.elements[0]
    # Does the test case actually different values for evaluation at r =
    # (0, 0, 0) vs. averaging over the Gauss points?
    σ_center = e.tstress((0, 0, 0))
    σ_gpt = np.mean([e.tstress(r) for r in e.gloc], axis=0)
    assert np.all(np.abs(σ_gpt - σ_center) > 0.003)
    # Does waffleiron' stress averaged over Gauss points match FEBio
    # output stress?
    σ_FEBio = model.solution.value("stress", -1, 1, 1)
    npt.assert_allclose(σ_gpt, σ_FEBio, rtol=RTOL_STRESS, atol=ATOL_STRESS)


class ElementMethodsTestQuad4(unittest.TestCase):
    def setUp(self):
        self.model = wfl.load_model(DIR_FIXTURES / "center-crack-2d-1mm.feb")


class FTestTri3(unittest.TestCase):
    """Test F tensor calculations for Tri3 mesh.

    Only part of the F tensor is tested right now, pending full
    implementation of the extended directors.

    """

    def setUp(self):
        self.model = wfl.load_model(DIR_FIXTURES / "square_tri3.feb")
        self.elemdata = wfl.input.textdata_list(
            os.path.join("test", "fixtures", "square_tri3_elem_data.txt"), delim=","
        )

    def test_f(self):
        istep = -1
        u = self.model.solution.step_data(istep)[("displacement", "node")]
        for eid in range(len(self.model.mesh.elements)):
            F_expected = f_tensor_logfile(self.elemdata, istep, eid)
            F = self.model.mesh.elements[eid].f((1.0 / 3.0, 1.0 / 3.0))
            npt.assert_almost_equal(F[:2, :2], F_expected[:2, :2], decimal=5)


@unittest.skip(
    "extensible directors not yet implemented, so shell elements will not provide the correct F tensor"
)
class FTestQuad4(unittest.TestCase):
    """Test F tensor calculations for Tri3 mesh."""

    def setUp(self):
        self.soln = wfl.MeshSolution("test/fixtures/" "square_quad4.xplt")
        self.elemdata = wfl.textdata_list(
            "test/fixtures/" "square_quad4_elem_data.txt", delim=","
        )

    def test_f(self):
        istep = -1
        u = self.soln.reader.step_data(istep)[("displacement", "node")]
        for eid in range(len(self.soln.elements)):
            F_expected = f_tensor_logfile(self.elemdata, istep, eid)
            F = self.soln.elements[eid].f((1.0 / 3.0, 1.0 / 3.0), u)
            npt.assert_almost_equal(F[:2, :2], F_expected[:2, :2], decimal=5)


def test_integration():
    # create trapezoidal element
    nodes = ((0.0, 0.0), (2.0, 0.0), (1.5, 2.0), (0.5, 2.0))
    element = wfl.element.Quad4(nodes)
    # compute area
    actual = element.integrate(lambda e, r: 1.0)
    desired = 3.0  # A_trapezoid = 0.5 * (b1 + b2) * h
    npt.assert_approx_equal(actual, desired)


def test_dinterp_2d():
    """Test dinterp with a truly 2D element."""
    # create square element
    nodes = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    element = wfl.element.Quad4(nodes)
    element.properties["testval"] = np.array((0.0, 10.0, 11.0, 1.0))
    desired = np.array([10.0, 1.0])
    actual = element.dinterp((0, 0), "testval").reshape(-1)
    npt.assert_allclose(actual, desired)

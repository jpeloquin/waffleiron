# Python built-ins
from math import radians, sin, cos
import os
from pathlib import Path

# Public packages
import numpy as np
import pytest

# waffleiron' local modules
import waffleiron as wfl
from waffleiron.material import from_Lamé


RTOL_F = 5e-7
ATOL_F = 5e-8
RTOL_STRESS = 5e-7
ATOL_STRESS = 5e-7
DIR_FIXTURES = Path(__file__).parent / "fixtures"
DIR_OUT = Path(__file__).parent / "output"
if not DIR_OUT.exists():
    DIR_OUT.mkdir()


@pytest.fixture
def pytest_request(request):
    return request


# Versions against which to test.

# All tests that run FEBio should use every supported FEBio version.
FEBIO_CMDS = ("febio2", "febio3.3", "febio4")


@pytest.fixture(scope="session", params=FEBIO_CMDS)
def febio_cmd(request):
    """Run test with all supported FEBio versions"""
    return request.param


# All tests that write FEBio XML should use every supported FEBio XML
# version.
FEBIO_XMLVERS = ("2.5", "3.0")


@pytest.fixture(scope="session", params=FEBIO_XMLVERS)
def xml_version(request):
    """Run test with all supported FEBio XML versions"""
    return request.param


# Tests that both write FEBio XML *and* run FEBio should use these combinations of
# FEBio version and FEBio XML version, as FEBio 2 cannot read FEBio XML 3.0.
FEBIO_CMDS_XMLVERS = (
    ("febio2", "2.5"),
    ("febio3", "2.5"),
    ("febio3", "3.0"),  # 3.3.3 or 3.8+, since 3.4–3.7 doesn't have node reaction force
    ("febio4", "3.0"),
    ("febio4", "4.0"),
)
# TODO: Figure out some way to filter test parameters by major and minor version,
#  since some FEBio versions support different things.


@pytest.fixture(
    scope="session",
    params=FEBIO_CMDS_XMLVERS,
    ids=(f"{c},xml{v}" for c, v in FEBIO_CMDS_XMLVERS),
)
def febio_cmd_xml(request):
    """Run test with all supported combinations of FEBio and FEBio XML"""
    cmd, xml_version = request.param
    return cmd, xml_version


@pytest.fixture(
    scope="session",
    params=[(c, v) for c, v in FEBIO_CMDS_XMLVERS if c != "febio2"],
    ids=[f"{c},xml{v}" for c, v in FEBIO_CMDS_XMLVERS if c != "febio2"],
)
def febio_3plus_cmd_xml(request):
    """Run test with all supported combinations of FEBio and FEBio XML"""
    cmd, xml_version = request.param
    return cmd, xml_version


@pytest.fixture(
    scope="session",
    params=[(c, v) for c, v in FEBIO_CMDS_XMLVERS if c not in ("febio2", "febio3")],
    ids=[f"{c},xml{v}" for c, v in FEBIO_CMDS_XMLVERS if c not in ("febio2", "febio3")],
)
def febio_4plus_cmd_xml(request):
    """Run test with all supported combinations of FEBio and FEBio XML"""
    cmd, xml_version = request.param
    return cmd, xml_version


F_monoaxial = (
    np.diag([1.08, 1, 1]),
    np.diag([1, 1.08, 1]),
    np.diag([1, 1, 1.08]),
    np.diag([0.92, 1, 1]),
    np.diag([1, 0.94, 1]),
    np.diag([1, 1, 0.94]),
)
F_multiaxial = (
    np.diag([1.02, 1.03, 1.04]),
    np.diag([0.98, 1.04, 1.03]),
    np.diag([1.04, 0.96, 1.03]),
    np.diag([1.04, 1.02, 0.97]),
    np.diag([0.98, 0.97, 1.02]),
    np.diag([0.98, 1.02, 0.97]),
    np.diag([1.02, 0.97, 0.96]),
    np.diag([0.96, 0.97, 0.98]),
)
F_shear = (
    np.array([[1, 0.02, 0.03], [-0.022, 1, 0.05], [-0.033, -0.044, 1]]),
    np.array([[1, -0.02, -0.03], [0.022, 1, -0.05], [0.033, 0.044, 1]]),
)
F_rotations = (
    np.array(
        [
            [1, 0, 0],
            [0, cos(radians(40)), -sin(radians(40))],
            [0, sin(radians(40)), cos(radians(40))],
        ]
    ),
    np.array(
        [
            [cos(radians(30)), 0, sin(radians(30))],
            [0, 1, 0],
            [-sin(radians(30)), 0, cos(radians(30))],
        ]
    ),
    np.array(
        [
            [cos(radians(25)), -sin(radians(25)), 0],
            [sin(radians(25)), cos(radians(25)), 0],
            [0, 0, 1],
        ]
    ),
)


def gen_model_center_crack_Hex8():
    """A 10 mm × 20 mm rectangle with a center 2 mm crack.

    Material: isotropic linear elastic.

    Boundary conditions: 2% strain applied in y.

    """
    model = wfl.load_model(
        DIR_FIXTURES / "center_crack_uniax_isotropic_elastic_hex8.feb"
    )

    material = model.mesh.elements[0].material
    γ = material.λ
    μ = material.μ
    E, ν = from_Lamé(γ, μ)

    crack_line = ((-0.001, 0.0, 0.0), (0.001, 0.0, 0.0))

    # right tip
    tip_line_r = [
        i
        for i, (x, y, z) in enumerate(model.mesh.nodes)
        if (np.allclose(x, crack_line[1][0]) and np.allclose(y, crack_line[1][1]))
    ]
    tip_line_r = set(tip_line_r)

    tip_line_l = [
        i
        for i, (x, y, z) in enumerate(model.mesh.nodes)
        if (np.allclose(x, crack_line[0][0]) and np.allclose(y, crack_line[0][1]))
    ]
    tip_line_l = set(tip_line_l)

    # identify crack faces
    f_candidates = wfl.select.surface_faces(model.mesh)
    f_seed = [f for f in f_candidates if (len(set(f) & tip_line_r) > 1)]
    f_crack_surf = wfl.select.f_grow_to_edge(f_seed, model.mesh)
    crack_faces = f_crack_surf

    attrib = {
        "E": E,
        "ν": ν,
        "tip_line_l": tip_line_l,
        "tip_line_r": tip_line_r,
        "crack_line": crack_line,
        "crack_faces": crack_faces,
    }

    return model, attrib


def gen_model_single_spiky_Hex8(material=None):
    """Return a model consisting of a single spiky Hex8 element.

    None of the edges of the Hex8 element are parallel to each other.
    The element is intended to be used as a fixture in tests of
    element-local basis vectors or issues related to spatial variation
    in shape function interpolation within the element.

    Material: Isotropic linear elastic.

    Boundary conditions: None.

    """
    # Create an irregularly shaped element in which no two edges are
    # parallel.
    x1 = np.array([0, 0, 0])
    x2 = [
        np.cos(radians(17)) * np.cos(radians(6)),
        np.sin(radians(17)) * np.cos(radians(6)),
        np.sin(radians(6)),
    ]
    x3 = np.array([0.8348, 0.9758, 0.3460])
    x4 = np.array([0.0794, 0.9076, 0.1564])
    x5 = x1 + np.array(
        [
            0.638 * np.cos(radians(26)) * np.sin(radians(1)),
            0.638 * np.sin(radians(26)) * np.sin(radians(1)),
            np.cos(radians(1)),
        ]
    )
    x6 = x5 + np.array(
        [
            0.71 * np.cos(radians(-24)) * np.cos(radians(-7)),
            0.71 * np.sin(radians(-24)) * np.cos(radians(-7)),
            np.sin(radians(-7)),
        ]
    )
    x7 = [1, 1, 1]
    x8 = x5 + [
        np.sin(radians(9)) * np.cos(radians(-11)),
        np.cos(radians(9)) * np.cos(radians(-11)),
        np.sin(radians(-11)),
    ]
    nodes = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])
    element = wfl.element.Hex8.from_ids([i for i in range(8)], nodes)
    element.material = material
    model = wfl.Model(wfl.Mesh(nodes, [element]))
    return model

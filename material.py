from functools import partial
from math import log, exp, sin, cos, acos, radians, pi, inf
from typing import Callable, Tuple

import chaospy  # TODO: just need quadrature points
import numpy as np
from numpy import dot, trace, eye, outer
from numpy.linalg import det

# Same-package modules
from .core import CONSTANT_R, CONSTANT_F, Sequence, ScaledSequence
from .exceptions import InvalidParameterError

_DEFAULT_ORIENT_RANK1 = np.array([1, 0, 0])

_DEFAULT_ORIENT_RANK2 = (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

# Precalculated ellipsoidal fiber orientation distribution integration terms from
# FEBio geodesic.h.  Each row is cos(Θ) * sin(φ), sin(Θ) * sin(φ), cos(φ), weight;
# θ ∈ [0, π/2], φ ∈ [0, π/2].  The weights sum to π/2.
# TODO: double precision
FIBER_OCTANT_INTEGRATION_WEIGHTS = np.array(
    [
        [1, 0, 0, 0.003394024],
        [0, 1, 0, 0.003394024],
        [0, 0, 1, 0.003394024],
        [0.7071068, 0.7071068, 0, 0.02550091],
        [0, 0.7071068, 0.7071068, 0.02550091],
        [0.7071068, 0, 0.7071068, 0.02550091],
        [0.9486833, 0.3162278, 0, 0.0180476],
        [0.3162278, 0.9486833, 0, 0.0180476],
        [0, 0.9486833, 0.3162278, 0.0180476],
        [0, 0.3162278, 0.9486833, 0.0180476],
        [0.9486833, 0, 0.3162278, 0.0180476],
        [0.3162278, 0, 0.9486833, 0.0180476],
        [0.4082483, 0.8164966, 0.4082483, 0.06535968],
        [0.4082483, 0.4082483, 0.8164966, 0.06535968],
        [0.8164966, 0.4082483, 0.4082483, 0.06535968],
        [0.9899495, 0.1414214, 0, 0.01273219],
        [0.8574929, 0.5144958, 0, 0.02322682],
        [0.5144958, 0.8574929, 0, 0.02322682],
        [0.1414214, 0.9899495, 0, 0.01273219],
        [0, 0.9899495, 0.1414214, 0.01273219],
        [0, 0.8574929, 0.5144958, 0.02322682],
        [0, 0.5144958, 0.8574929, 0.02322682],
        [0, 0.1414214, 0.9899495, 0.01273219],
        [0.9899495, 0, 0.1414214, 0.01273219],
        [0.8574929, 0, 0.5144958, 0.02322682],
        [0.5144958, 0, 0.8574929, 0.02322682],
        [0.1414214, 0, 0.9899495, 0.01273219],
        [0.5883484, 0.7844645, 0.1961161, 0.05866665],
        [0.1961161, 0.7844645, 0.5883484, 0.05866665],
        [0.5883484, 0.1961161, 0.7844645, 0.05866665],
        [0.1961161, 0.5883484, 0.7844645, 0.05866665],
        [0.7844645, 0.5883484, 0.1961161, 0.05866665],
        [0.7844645, 0.1961161, 0.5883484, 0.05866665],
        [0.9128709, 0.3651484, 0.1825742, 0.04814243],
        [0.9128709, 0.1825742, 0.3651484, 0.04814243],
        [0.9733285, 0.1622214, 0.1622214, 0.03438731],
        [0.1622214, 0.9733285, 0.1622214, 0.03438731],
        [0.1825742, 0.9128709, 0.3651484, 0.04814243],
        [0.3651484, 0.9128709, 0.1825742, 0.04814243],
        [0.1825742, 0.3651484, 0.9128709, 0.04814243],
        [0.1622214, 0.1622214, 0.9733285, 0.03438731],
        [0.3651484, 0.1825742, 0.9128709, 0.04814243],
        [0.6396021, 0.4264014, 0.6396021, 0.07332545],
        [0.4264014, 0.6396021, 0.6396021, 0.07332545],
        [0.6396021, 0.6396021, 0.4264014, 0.07332545],
    ]
)


def pdf3d_spherical():
    return 1 / 4 / pi


def pdf3d_ellipsoidal(r: np.ndarray, d: np.ndarray):
    # TODO: normalize so this is actually a pdf
    # return ((r[0] / d[0]) ** 2 + (r[1] / d[1]) ** 2 + (r[2] / d[2]) ** 2) ** -0.5
    return 1 / np.linalg.norm(r / d)


def integrate_sph2_oct(summand, f):
    """Integrate f over unit half-sphere

    45 points per octant

    """
    for octant_signs in (
        np.array([1, 1, 1]),
        np.array([-1, 1, 1]),
        np.array([-1, -1, 1]),
        np.array([1, -1, 1]),
    ):
        for i in range(FIBER_OCTANT_INTEGRATION_WEIGHTS.shape[0]):
            N = octant_signs * FIBER_OCTANT_INTEGRATION_WEIGHTS[i, :3]
            w = FIBER_OCTANT_INTEGRATION_WEIGHTS[i, -1]
            summand += w * f(N)
    return summand


def integrate_sph2_gkt(summand, f, o_φ, n_θ):
    """Integrate f over unit half-sphere by GKT

    Implemented based on Hou_Ateshian_2016, but integrating the whole sphere.

    """
    # FEBio weights sum to 2; these weights sum to 1.  That doesn't affect the FEBio
    # results because they normalize by the integrated fiber density.
    ζ_points, ζ_weights = chaospy.quadrature.kronrod(o_φ, chaospy.Uniform(-1, 1))
    # ^ takes 2.2 ms; maybe call it once (cache it?)  Whole integration takes 3 ms.
    ζ_points = ζ_points.squeeze()
    ζ_weights = ζ_weights.squeeze()

    def φ_from_ζ(ζ):
        return acos(0.5 * (ζ * (cos(φb) - cos(φa)) + cos(φa) + cos(φb)))

    # θ is azimuth
    # φ is declination
    dθ = 2 * pi / n_θ
    φa = 0
    φb = pi / 2
    # ζa = -1
    # ζb = 1
    # h_ζ = 0.5 * (ζb - ζa)
    # center_ζ = 0.5 * (ζb + ζa)
    for ζ, w in zip(ζ_points, ζ_weights):
        for i in range(n_θ):
            θ = i * dθ
            φ = φ_from_ζ(ζ)
            N = np.array([cos(θ) * sin(φ), sin(θ) * sin(φ), cos(φ)])
            # np.testing.assert_almost_equal(np.linalg.norm(N), 1, decimal=14)
            summand += w * dθ * f(N)
    return summand


def deviatoric_stress(σ_tilde):
    """Return deviatoric stress from σ_tilde"""
    # The np.trace(σ_tilde) / 3 * np.eye(3) forces the stress to be deviatoric.
    # It feels like a properly constructed deviatoric constitutive equation
    # should already produce deviatoric stress.  Simply discarding the
    # hydrostatic part feels wrong.
    return σ_tilde - np.trace(σ_tilde) / 3 * np.eye(3)


def stress_1d_N(F, stress: Callable, N, **kwargs):
    """Return stress along (material config) unit vector N

    :param F: Deformation gradient tensor.

    :param stress: Function of stretch ratio λ that returns 1D stress; scalar → scalar.

    :param N: Unit vector along which to calculate stress.  N is defined in the
    (unstrained) material configuration.  N is not renormalized during the calculation,
    so if it is not a unit vector you will get incorrect results.

    """
    λ = np.linalg.norm(F @ N)  # np.sqrt(Q @ F.T @ F @ Q.T)
    P = stress(λ, **kwargs) * np.outer(N, N)  # PK2 stress
    J = np.linalg.det(F)
    σ = 1 / J * F @ P @ F.T
    return σ


def to_Lamé(E, v):
    """Convert Young's modulus & Poisson ratio to Lamé parameters."""
    if v <= -1 or v >= 0.5:
        raise InvalidParameterError(
            f"ν = {v} cannot be converted to Lamé parameters; -1 < ν < 0.5 required."
        )
    y = v * E / ((1.0 + v) * (1.0 - 2.0 * v))  # TODO: handle ν = 0.5 or -1
    mu = E / (2.0 * (1.0 + v))
    return y, mu


def from_Lamé(y, u):
    """Convert Lamé parameters to modulus & Poisson's ratio."""
    E = u / (y + u) * (2.0 * u + 3.0 * y)
    v = 0.5 * y / (y + u)
    return E, v


def orthotropic_elastic_compliance_matrix(material):
    """Return stiffness matrix for an orthotropic material with standard E, G, ν"""
    E1 = material.E1
    E2 = material.E2
    E3 = material.E3
    G12 = material.G12
    G23 = material.G23
    G31 = material.G31
    ν12 = material.ν12
    ν13 = material.ν31 * material.E1 / material.E3
    ν21 = material.ν12 * material.E2 / material.E1
    ν23 = material.ν23
    ν31 = material.ν31
    ν32 = material.ν23 * material.E3 / material.E2
    S = np.array(
        [
            [1 / E1, -ν21 / E2, -ν31 / E3, 0, 0, 0],
            [-ν12 / E1, 1 / E2, -ν32 / E3, 0, 0, 0],
            [-ν13 / E1, -ν23 / E2, 1 / E3, 0, 0, 0],
            [0, 0, 0, 1 / G12, 0, 0],
            [0, 0, 0, 0, 1 / G23, 0],
            [0, 0, 0, 0, 0, 1 / G31],
        ]
    )
    return S


def orthotropic_elastic_stiffness_matrix(material):
    """Return stiffness matrix for an orthotropic material with standard E, G, ν"""
    E1 = material.E1
    E2 = material.E2
    E3 = material.E3
    G12 = material.G12
    G23 = material.G23
    G31 = material.G31
    ν12 = material.ν12
    ν13 = material.ν31 * material.E1 / material.E3
    ν21 = material.ν12 * material.E2 / material.E1
    ν23 = material.ν23
    ν31 = material.ν31
    ν32 = material.ν23 * material.E3 / material.E2
    a = 1 - ν12 * ν21 - ν23 * ν32 - ν31 * ν13 - 2 * ν12 * ν23 * ν31
    C = np.array(
        [
            [
                (1 - ν23 * ν32) * E1 / a,
                (ν21 + ν31 * ν23) * E1 / a,
                (ν31 + ν21 * ν32) * E1 / a,
                0,
                0,
                0,
            ],
            [
                (ν12 + ν13 * ν32) * E2 / a,
                (1 - ν31 * ν13) * E2 / a,
                (ν32 + ν31 * ν12) * E2 / a,
                0,
                0,
                0,
            ],
            [
                (ν13 + ν12 * ν23) * E3 / a,
                (ν23 + ν13 * ν21) * E3 / a,
                (1 - ν21 * ν12) * E3 / a,
                0,
                0,
                0,
            ],
            [0, 0, 0, G12, 0, 0],
            [0, 0, 0, 0, G23, 0],
            [0, 0, 0, 0, 0, G31],
        ]
    )
    return C


def is_positive_definite(A):
    """Return True if matrix A is positive definite"""
    if len(A.shape) != 2:
        return ValueError("Matrix must have two index dimensions.")
    if A.shape[0] != A.shape[1]:
        return ValueError("Matrix must be square")
    for i in range(1, A.shape[0] + 1):
        if np.linalg.det(A[:i, :i]) <= 0:
            return False
    return True


def unit_step(x):
    """Unit step function."""
    if x > 0.0:
        return 1.0
    else:
        return 0.0


def _is_fixed_property(p):
    if isinstance(p, Sequence) or isinstance(p, ScaledSequence):
        return False
    else:
        return True


class Material:
    """Base class for materials"""

    bounds = {}  # stub

    # TODO: Make bounds checking automatic, and make all materials inherit.

    def check_parameters_bounds(self):
        for k in self.bounds:
            v = getattr(self, k)
            if not self.bounds[k][0] < v < self.bounds[k][1]:
                raise InvalidParameterError(
                    f"{k} = {v} must be within {self.bounds[k]}"
                )


class Uncoupled:
    """Mixin class for uncoupled materials

    Classes inheriting from `Uncoupled` need only define `tilde_stress(F, **kwargs)`.
    The other stress functions will be provided by `Uncoupled`.

    """

    def tilde_stress(self, F, **kwargs):
        """Return σ_tilde stress

        Cauchy stress σ = dev(σ_tilde).  σ_tilde gets a separate function to support
        use of the material both alone and in a deviatoric mixture.

        This is a stub; override it when subclassing.

        """
        raise NotImplementedError

    def tstress(self, F, **kwargs):
        """Cauchy stress tensor"""
        return deviatoric_stress(self.tilde_stress(F, **kwargs))


class D1:
    """Marks a 1-dimensional material or fiber"""

    pass


class D3:
    """Marks a 3-dimensional material"""

    pass


class OrientedMaterial:
    """A material with an orientation matrix"""

    def __init__(self, material, Q=np.eye(3)):
        self.material = material
        Q = np.array(Q)
        Q = Q / np.linalg.norm(Q, axis=0)
        if isinstance(material, D1):
            if Q.ndim != 1:
                raise ValueError(
                    f"1D materials must have an R3 vector orientation.  Got {Q}."
                )
        elif isinstance(material, D3):
            if Q.ndim == 1:
                # Inflate orientation matrix to 3D
                e1 = Q
                i = np.where(Q)[0][0]
                j = (i + 1) % len(e1)
                e2 = np.zeros(3)
                e2[j] = -e1[i]
                e2[i] = e1[j]
                e2 = e2 / np.linalg.norm(e2)
                e3 = np.linalg.cross(e1, e2)
                Q = np.stack([e1, e2, e3])
            else:
                if Q.ndim != 2:
                    raise ValueError(
                        f"A 3D material must have a 3x3 orientation matrix.  Got {Q}."
                    )
        else:
            raise ValueError(
                f"{type(material)} must inherit from D1 or D3, so it has a known dimensionality, to become an oriented material."
            )
        self.orientation = Q

    def w(self, F):
        return self.material.w(F)

    def tstress(self, F, **kwargs):
        Q = self.orientation
        if Q.ndim == 1:
            # 1D material ("fiber")
            return stress_1d_N(F, self.material.stress, Q)
        elif Q.ndim == 2:
            # 3D material ("solid")
            σ_loc = self.material.tstress(F @ Q, **kwargs)
            # ^ Stress in own local basis.  This is a change of coordinate system for
            # the material, not an observer change, such that material anisotropy is
            # accounted for.
        else:
            raise ValueError(
                f"Orientation matrix should be 1st or 2nd order, not {Q.ndim}"
            )
        return σ_loc

    def pstress(self, F, **kwargs):
        Q = self.orientation
        return Q @ self.material.pstress(Q.T @ F)  # TODO: Check

    def sstress(self, F, **kwargs):
        Q = self.orientation
        if Q.ndim == 1:
            # 1D material ("fiber")
            N = Q
            λ = np.linalg.norm(F @ Q)
            s_loc = self.material.stress(λ) * np.outer(N, N)
        elif Q.ndim == 2:
            raise NotImplementedError  # Needs test case
            # 3D material ("solid")
            s_loc = Q @ self.material.sstress(Q.T @ F) @ Q.T  # TODO: Check
        else:
            raise ValueError
        return s_loc


class DeviatoricFiber(Material, Uncoupled, D3):

    def __init__(self, fiber, Q=np.array([1, 0, 0])):
        self.material = fiber
        Q = np.array(Q)
        if not Q.ndim == 1:
            raise ValueError(f"Fiber orientation must be a direction vector.  Got {Q}.")
        self.orientation = Q

    def tilde_stress(self, F, **kwargs):
        F_tilde = np.linalg.det(F) ** (-1 / 3) * F
        N = self.orientation
        λ_tilde = np.linalg.norm(F_tilde @ N)
        σ_tilde = (
            F_tilde
            @ (self.material.stress(λ_tilde) * np.outer(N, N))
            @ F_tilde.T
            / np.linalg.det(F)
        )
        return σ_tilde

    def __getattr__(self, item):
        # Intended so parameters of the underlying fiber material can be easily
        # accessed.  Make sure any functions that do calculation using deviatoric
        # strain are defined on DeviatoricFiber.
        return getattr(self.material, item)


class EllipsoidalDistribution(D3):

    def __init__(self, d, mat_fiber):
        """Return fibers with ellipsoidal orientation distribution

        a, b, and c are not independent; their ratios matter, their scale does not.

        FEBio XML type attribute = "ellipsoidal".

        """
        self.d = np.array(d)
        self.fiber = mat_fiber
        # TODO: not sure integration scheme belongs here
        self.integration = ("fibers-3d-gkt", 11, 31)  # max needed in Hou_Ateshian_2016
        self.o_φ = (self.integration[1] - 1) // 2
        self.n_θ = self.integration[2]

    def tstress(self, F, **kwargs):

        def σ(N):
            """Return fiber direction stress"""
            return stress_1d_N(F, self.fiber.stress, N)

        R = partial(pdf3d_ellipsoidal, d=self.d)
        integrated_density = integrate_sph2_gkt(0, R, self.o_φ, self.n_θ)
        σ = (
            integrate_sph2_gkt(
                np.zeros((3, 3)), lambda N: R(N) * σ(N), self.o_φ, self.n_θ
            )
        ) / integrated_density
        return σ


class Permeability:
    """Parent type for Permeability implementations."""

    def __init__(self, **kwargs):
        msg = """The Permeability class is meant only to serve as a supertype for
child classes that implement the necessary functionality.  Only its
child classes should be instantiated as objects."""
        raise NotImplementedError(msg)


class IsotropicConstantPermeability(Permeability):
    """Isotropic strain-independent permeability"""

    bounds = {"k": (0, inf)}

    def __init__(self, k, **kwargs):
        self.k = k

    @classmethod
    def from_feb(cls, perm, **kwargs):
        return cls(perm)


class IsotropicExponentialPermeability(Material, Permeability):
    """Isotropic exponential permeability"""

    bounds = {"k0": (0, inf), "M": (0, inf)}

    def __init__(self, k0, M, **kwargs):
        self.k0 = k0
        self.M = M


class IsotropicHolmesMowPermeability(Material, Permeability):
    """Isotropic Holmes-Mow permeability"""

    # The strain-free solid volume fraction also appears in PoroelasticSolid.  Keeping
    # them consistent is currently left up to the user.  Should their consistency be
    # enforced?  Usually the material is not updated in-place and may as well be
    # immutable.
    bounds = {
        "k0": (0, inf),
        "M": (0, inf),
        "α": (0, inf),
        "φ0_s": (0, inf),
    }

    def __init__(self, k0, M, α, φ0_s, **kwargs):
        self.k0 = k0
        self.M = M
        self.α = α
        self.φ0_s = φ0_s

    @classmethod
    def from_feb(cls, perm, M, alpha, phi0, **kwargs):
        # φ0_s is not included in the FEBio <permeability> element, but is nonetheless a
        # parameter of the permeability equation.  Handling this discrepancy has to be
        # done in the FEBio XMl parsing phase.
        return cls(perm, M, alpha, phi0)


class TransIsoHolmesMowPermeability(Material, Permeability):
    """Transversely isotropic Holmes-Mow permeability

    "perm-ref-trans-iso" in FEBio.

    """

    # The strain-free solid volume fraction also appears in PoroelasticSolid.  Keeping
    # them consistent is currently left up to the user.  Should their consistency be
    # enforced?  Usually the material is not updated in-place and may as well be
    # immutable.
    bounds = {
        "k0": (0, inf),  # cannot be zero
        "M0": (0, inf),
        "α0": (0, inf),
        "k1a": (0, inf),  # *can* be zero
        "k2a": (0, inf),  # *can* be zero
        "Ma": (0, inf),
        "αa": (0, inf),
        "k1t": (0, inf),  # *can* be zero
        "k2t": (0, inf),  # *can* be zero
        "Mt": (0, inf),
        "αt": (0, inf),
        "φ0_s": (0, inf),
    }

    def __init__(self, k0, M0, α0, k1a, k2a, Ma, αa, k1t, k2t, Mt, αt, φ0_s, **kwargs):
        self.k0 = k0
        self.M0 = M0
        self.α0 = α0
        self.k1a = k1a
        self.k2a = k2a
        self.Ma = Ma
        self.αa = αa
        self.k1t = k1t
        self.k2t = k2t
        self.Mt = Mt
        self.αt = αt
        self.φ0_s = φ0_s


class PoroelasticSolid:
    """Fluid-saturated solid."""

    def __init__(
        self, solid, permeability: Permeability, solid_fraction, fluid_density=0
    ):
        """Return PoroelasticSolid instance

        solid := Solid material instance.

        permeability := Permeability instance.

        solid_fraction := Volume fraction of solid.  Volume fraciton of
        solid + volume fraction of fluid = 1.

        """
        self.fluid_density = fluid_density
        self.solid_material = solid
        self.solid_fraction = solid_fraction
        if not isinstance(permeability, Permeability):
            # If the value is not a Permeability instance and is valid, it must be a
            # number, implicitly assuming isotropic constant permeability
            permeability = IsotropicConstantPermeability(permeability)
        self.permeability = permeability


class DonnanSwelling(Material, D3):
    """Swelling pressure of the Donnan equilibrium type."""

    bounds = {
        "fcd0": (0, inf),
        "phi0_w": (0, 1),  # open interval
        "ext_osm": (0, inf),
        "osm_coeff": (0, 1),
    }

    def __init__(self, phi0_w, fcd0, ext_osm, osm_coef, **kwargs):
        # Bounds checks
        if _is_fixed_property(phi0_w) and not (0 <= phi0_w <= 1):
            raise InvalidParameterError(
                f"phi0_w = {phi0_w}; it is required that 0 ≤ phi0_w ≤ 1"
            )
        if _is_fixed_property(fcd0) and not (fcd0 >= 0):
            raise InvalidParameterError(f"fcd0 = {fcd0}; it is required that 0 < fcd0")
        if _is_fixed_property(ext_osm) and not (ext_osm >= 0):
            raise InvalidParameterError(
                f"ext_osm = {ext_osm}; it is required that 0 < ext_osm"
            )
        # Store values
        self.phi0_w = phi0_w
        self.fcd0 = fcd0
        self.ext_osm = ext_osm
        self.osm_coef = osm_coef

    @classmethod
    def from_feb(cls, phiw0, cF0, bosm, Phi=1, **kwargs):
        return cls(phiw0, cF0, bosm, Phi)

    # TODO: find a way to pass T
    def tstress(self, F, T, R=CONSTANT_R, **kwargs):
        """Return Cauchy stress tensor"""
        # TODO: R units are going to be a constant source of bugs in user code until
        # waffleiron is fully units-aware
        J = np.linalg.det(F)
        FCD = self.phi0_w / (J - 1 + self.phi0_w) * self.fcd0
        p = R * T * self.osm_coef * ((FCD**2 + self.ext_osm**2) ** 0.5 - self.ext_osm)
        return -p * np.eye(3)


class Multigeneration:
    """Mixture of materials created at and referenced to a given time."""

    def __init__(self, generations, **kwargs):
        """Return Multigeneration object.

        generations := a list of tuples (start_time <float>, material
        <Material>), each tuple defining a material created at and
        referenced to `start_time`.

        """
        t, materials = zip(*generations)
        self.generation_times = t
        self.materials = materials


class SolidMixture(D3):
    """Mixture of solids with no interdependencies or residual stress.

    The strain energy of the mixture is defined as the sum of the strain energies for
    each component.

    """

    def __init__(self, solids, **kwargs):
        """Mixture of elastic solids

        :param solids: List of material instances comprising the mixture.

        """
        if not solids:
            raise ValueError(
                "SolidMixture requires at least one solid, but none were provided."
            )
        self.materials = [m for m in solids]

    def w(self, F):
        return sum(material.w(F) for material in self.materials)

    def tstress(self, F, *args, **kwargs):
        return sum(material.tstress(F, *args, **kwargs) for material in self.materials)

    def pstress(self, F, *args, **kwargs):
        return sum(material.pstress(F, *args, **kwargs) for material in self.materials)

    def sstress(self, F, *args, **kwargs):
        return sum(
            [material.sstress(F, *args, **kwargs) for material in self.materials]
        )


class DeviatoricSolidMixture(D3, Uncoupled):
    """Mixture of solids with no interdependencies or residual stress.

    The strain energy of the mixture is defined as the sum of the strain energies for
    each component.

    """

    def __init__(self, solids, **kwargs):
        """Mixture of elastic solids

        :param solids: List of uncoupled material instances comprising the mixture.

        """
        if not solids:
            raise ValueError(
                "DeviatoricSolidMixture requires at least one solid, but none were provided."
            )
        self.materials = [m for m in solids]

    def tstress(self, F, *args, **kwargs):
        σ = deviatoric_stress(
            sum(
                material.tilde_stress(F, *args, **kwargs) for material in self.materials
            )
        )
        return σ


class Rigid:
    """Pseudo-material used for elements in rigid bodies"""

    def __init__(self, props={}, **kwargs):
        self.density = 0
        if "density" in props:
            self.density = props["density"]


class NeoHookeanFiber(D1):
    """1D fiber with σ ~ λ^2 − 1 relation

    Same as "fiber-NH" in FEBio.

    """

    bounds = {
        "E": (0, inf),
    }

    def __init__(self, E):
        self.E = E

    def stress(self, λ):
        """Return fiber stress scalar along original orientation

        If embedding in R3, treat this as 2nd Piola–Kirchoff stress.

        """
        if λ <= 1:
            return 0
        else:
            return self.E * (λ**2 - 1)


class NaturalNeoHookeanFiber(D1):
    """1D fiber with σ ~ ln(λ) / λ^2 relation

    Also called "natural neo-Hookean".

    Same as "fiber-natural-NH" in FEBio; available in FEBio ≥ 3.5.1 (2021-09-28).

    """

    bounds = {
        "E": (0, inf),
        "λ0": (1, inf),
    }

    def __init__(self, E, λ0):
        self.E = E
        self.λ0 = λ0

    def stress(self, λ):
        """Return fiber stress scalar along original orientation

        If embedding in R3, treat this as 2nd Piola–Kirchoff stress.

        """
        if λ <= self.λ0:
            return 0
        else:
            return self.E / λ**2 * np.log(λ / self.λ0)


class ExponentialFiber(D1):
    """1D fiber with exponential power law.

    Coupled formulation ("fiber-exp-pow" in FEBio < 3.5.1; more recent releases have
    λ0).

    References
    ----------
    FEBio users manual 2.9, page 144.

    """

    bounds = {
        "ξ": (0, inf),  # ξ > 0
        "α": (0, inf),  # α ≥ 0
        "β": (2, inf),  #  β ≥ 2
    }

    def __init__(self, ξ, α, β):
        self.ξ = ξ
        self.α = α
        self.β = β

    def w(self, λ):
        """Return pseudo-strain energy density."""
        # TODO: Unit test
        α = self.α
        β = self.β
        ξ = self.ξ
        Ψ = ξ / (α * β) * (exp(α * (λ**2 - 1) ** β) - 1)
        return Ψ

    def stress(self, λ):
        """Return material stress scalar."""
        ξ = self.ξ
        α = self.α
        β = self.β
        if λ <= 1:
            σ = 0
        else:
            with np.errstate(invalid="raise"):
                dΨ_dλsq = ξ * (λ**2 - 1) ** (β - 1) * exp(α * (λ**2 - 1) ** β)
            # Use dΨ_dλsq instead of dΨ_dλ because this is the equivalent of dΨ/dC.
            σ = 2 * dΨ_dλsq
        return σ

    def sstress(self, λ):
        """Synonym for material stress.

        Makes some use cases easier by providing a consistent name with
        3D materials.

        """
        return self.stress(λ)


class ExponentialFiber3D(D3):
    """Fiber with exponential power law.

    Coupled formulation ("fiber-exp-pow" in FEBio < 3.5.1; more recent releases
    have λ0).

    This is a deprecated 3D implementation that mixes material orientation with the
    fiber's constitutive law.  Prefer ExponentialFiber, which does not mix concerns in
    this manner.

    References
    ----------
    FEBio users manual 2.0, page 104.

    """

    def __init__(self, ξ, α, β, orientation=_DEFAULT_ORIENT_RANK1, **kwargs):
        self.ξ = ξ
        self.α = α
        self.β = β
        self.orientation = orientation

    def w(self, F):
        """Return strain energy density."""
        F = np.array(F)

        # deviatoric components
        # J = det(F)
        # Fdev = J**(-1.0/3.0) * F
        # Cdev = dot(Fdev.T, Fdev)
        C = dot(F.T, F)
        # fiber unit vector
        N = self.orientation
        # square of fiber stretch
        In = dot(N, dot(C, N))
        a = self.α
        b = self.β
        xi = self.ξ
        w = xi / (a * b) * (exp(a * (In - 1.0) ** b) - 1.0)
        return w

    def tstress(self, F, **kwargs):
        """Return Cauchy stress tensor"""
        F = np.array(F)
        # Components
        J = det(F)
        C = dot(F.T, F)
        # fiber unit vector
        N = self.orientation
        # Square of fiber stretch
        In = dot(N, dot(C, N))
        yf = In**0.5  # fiber stretch
        n = dot(F, N) / yf

        a = self.α
        b = self.β
        xi = self.ξ
        dPsi_dIn = xi * (In - 1.0) ** (b - 1.0) * exp(a * (In - 1.0) ** b)
        t = (2 / J) * unit_step(In - 1.0) * In * dPsi_dIn * outer(n, n)
        return t

    def pstress(self, F, **kwargs):
        """Return 1st Piola–Kirchoff stress tensor"""
        t = self.tstress(F)
        p = det(F) * dot(t, np.linalg.inv(F).T)
        return p

    def sstress(self, F, **kwargs):
        """Return 2nd Piola-Kirchoff stress tensor"""
        t = self.tstress(F)
        s = det(F) * dot(np.linalg.inv(F), dot(t, np.linalg.inv(F).T))
        return s


class PowerLinearFiber(D1):
    """1D fiber with power–linear law.

    Equivalent to "fiber-pow-linear" or "fiber-power-linear" in FEBio, treated
    as a 1D material.

    References
    ----------
    FEBio users manual 2.9, page 173 (not page 146; the manual shows the
    wrong equations for the "fiber-pow-linear" entry).

    """

    bounds = {
        "E": (0, inf),
        "β": (2, inf),  # I don't see why FEBio has 2 is the minimum
        "λ0": (1, inf),  # FEBio has > 1; I don't see why λ0 = 1 is invalid
    }

    def __init__(self, E, β, λ0):
        self.E = E  # fiber modulus in linear region
        self.β = β  # power law exponent in power law region
        self.λ0 = λ0  # stretch ratio at which power law region
        # transitions to linear region

    @classmethod
    def from_feb(cls, E, beta, lam0, **kwargs):
        return cls(E, beta, lam0)

    def w(self, λ):
        """Return pseudo-strain energy density."""
        raise NotImplementedError

    def stress(self, λ):
        """Return stress scalar"""
        λ0 = self.λ0
        β = self.β
        E = self.E
        # Stress
        λsq = λ**2
        if λ <= 1:
            σ = 0
        elif λ <= λ0:
            ξ = E / 2 / (β - 1) * λ0 ** (-3) * (λ0**2 - 1) ** (2 - β)
            dΨ_dλsq = ξ / 2 * (λsq - 1) ** (β - 1)
            σ = 2 * dΨ_dλsq  # equiv to 2 dΨ/dC; PK2
        else:
            b = E / 2 / (λ0**3) * ((λ0**2 - 1) / (2 * (β - 1)) + λ0**2)
            dΨ_dλsq = b - E / 2 / λ
            σ = 2 * dΨ_dλsq  # equiv to 2 dΨ/dC; PK2
        return σ

    def sstress(self, λ):
        """Synonym for material stress.

        Makes some use cases easier by providing a consistent name with 3D materials.

        """
        return self.stress(λ)


class PowerLinearFiber3D(D3):
    """Fiber with piecewise power-law (toe) and linear regions.

    Coupled formulation ("fiber-pow-linear" or "fiber-power-linear" in FEBio).

    This is a deprecated 3D implementation that mixes material orientation with the
    fiber's constitutive law.  Prefer PowerLinearFiber, which does not mix concerns in
    this manner.

    """

    def __init__(self, E, β, λ0, orientation=_DEFAULT_ORIENT_RANK1, **kwargs):
        self.E = E  # fiber modulus in linear region
        self.β = β  # power law exponent in power law region
        self.λ0 = λ0  # stretch ratio at which power law region
        # transitions to linear region
        self.orientation = orientation

    def w(self, F):
        """Return strain energy density"""
        raise NotImplementedError

    def tstress(self, F, **kwargs):
        """Return Cauchy stress tensor aligned to local axes."""
        # Properties
        I0 = self.λ0**2.0
        β = self.β
        E = self.E
        N = self.orientation
        ξ = E / 4 / (β - 1) * I0 ** (-1.5) * (I0 - 1) ** (2 - β)
        b = E / 2 / np.sqrt(I0) + ξ * (I0 - 1) ** (β - 1)
        ψ0 = ξ / (2 * β) * (I0 - 1) ** β
        # Deformation
        C = F.T @ F
        I_N = N @ C @ N
        n = F @ N / np.sqrt(I_N)
        J = det(F)
        # Stress
        if I_N <= 1:
            σ = np.zeros((3, 3))
        elif I_N <= I0:
            # return 1 / J * ξ/2 * (I_N - 1)**(β - 1) * 2 * F @ F @ np.outer(N,N)
            σ = 2 / J * I_N * ξ * (I_N - 1) ** (β - 1) * np.outer(n, n)
        else:
            σ = 2 / J * I_N * (b - E / 2 / np.sqrt(I_N)) * np.outer(n, n)
        return σ

    def pstress(self, F, **kwargs):
        """Return 1st Piola–Kirchoff stress tensor aligned to local axes."""
        raise NotImplementedError

    def sstress(self, F, **kwargs):
        """Return 2nd Piola–Kirchoff stress tensor aligned to local axes."""
        raise NotImplementedError


class ExpAndLinearDCFiber(D1):
    """1D fiber with piecewise exponential–linear stress and discontinuous elasticity.

    Equivalent to "fiber-exp-linear" in FEBio, treated as a 1D material.

    """

    bounds = {
        "ξ": (0, inf),
        "α": (0, inf),
        "λ1": (1, inf),  # FEBio has > 1; I don't see why λ0 = 1 is invalid
        "E": (0, inf),  # linear modulus
    }

    def __init__(self, ξ, α, λ1, E):
        self.ξ = ξ  # overall coefficient
        self.α = α  # exponent's coefficient
        self.λ1 = λ1  # transition stretch ratio
        self.E = E  # linear modulus
        self.σ0 = self.ξ * (np.exp(self.α * (self.λ1 - 1)) - 1) - self.E * self.λ1

    def stress(self, λ, **kwargs):
        """Return stress scalar"""
        # Stress
        if λ <= 1:
            σ = 0
        elif λ <= self.λ1:
            σ = self.ξ / λ**2 * (np.exp(self.α * (λ - 1)) - 1)
        else:
            σ = self.E / λ + self.σ0 / λ**2
        return σ


class EllipsoidalPowerFiber(D3):
    """Power-law fibers with ellipsoidal orientation distribution.

    Coupled formulation.  FEBio XML name = "ellipsoidal fiber distribution".  In both
    FEBio and Waffleiron, this constitutive law should be part of the general-purpose
    fiber orientation distribution framework, but in FEBio it's a standalone special
    case with a unique constitutive law and a unique integeration scheme.

    """

    bounds = {
        "ξ": ((0, inf), (0, inf), (0, inf)),
        "β": ((0, inf), (0, inf), (0, inf)),
    }

    def __init__(self, ξ: Tuple[float, float, float], β: Tuple[float, float, float]):
        """Return EllipsoidalPowerFiber instance

        :param ξ: (ξx, ξy, ξz) tuple of directional fiber coefficients.

        :param β: (βx, βy, βz) tuple of direction fiber exponents.

        """
        self.ξ = ξ
        self.β = β

    @classmethod
    def from_feb(cls, ksi, beta):
        return cls(ksi, beta)

    def tstress(self, F, **kwargs):
        """Return Cauchy stress tensor

        :param F: Deformation gradient tensor.

        """
        J = np.linalg.det(F)

        def σ_N(N):
            """Return fiber direction stress

            :param N: Fiber unit vector in the reference configuration.

            Omit 2 / J factor to avoid repeated calculation; add it later.

            """
            v_n = F @ N  # directed fiber stretch ratio, deformed configuration
            λ_n = np.linalg.norm(v_n)  # fiber stretch ratio
            n = v_n / λ_n  # fiber unit vector, deformed configuration
            I_n = λ_n**2
            if λ_n <= 1:
                return np.zeros((3, 3))
            ξ = sum((N[i] / self.ξ[i]) ** 2 for i in range(3)) ** -0.5
            β = sum((N[i] / self.β[i]) ** 2 for i in range(3)) ** -0.5
            dΨ = β * ξ * (I_n - 1) ** (β - 1)
            return 2 * I_n / J * dΨ * np.outer(n, n)

        σ = integrate_sph2_oct(np.zeros((3, 3)), σ_N)
        # The literature integrates over the full sphere (Ateshian & Hung 2009).  I
        # think this is a strange choice because fibers are lines, not rays.
        # Integrating over the full sphere counts each fiber family twice.  Nevertheless
        # consistency with FEBio and the literature is probably best.
        return 2 * σ


class IsotropicElastic(D3):
    """Isotropic elastic material definition."""

    def __init__(self, props, **kwargs):
        if "E" in props and "v" in props:
            y, mu = to_Lamé(props["E"], props["v"])
        elif "lambda" in props and "mu" in props:
            y = props["lambda"]
            mu = props["mu"]
        else:
            raise Exception(
                "The combination of material properties "
                "in " + str(props) + " is not yet "
                "implemented."
            )
        self.mu = mu
        self.y = y

    def w(self, F):
        """Strain energy for isotropic elastic material.

        F = deformation tensor
        y = Lamé parameter λ
        mu = Lamé parameter μ

        """
        y = self.y
        mu = self.mu
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        W = 0.5 * y * trE**2.0 + mu * np.sum(E * E)
        return W

    def tstress(self, F, **kwargs):
        """Cauchy stress."""
        s = self.sstress(F)
        J = np.linalg.det(F)
        t = np.dot(np.dot(F, s), F.T) / J
        return t

    def pstress(self, F, **kwargs):
        """1st Piola-Kirchoff stress."""
        s = self.sstress(F)
        p = np.dot(s, F.T)
        return p

    def sstress(self, F, **kwargs):
        """2nd Piola-Kirchoff stress."""
        y = self.y
        mu = self.mu
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        s = 2.0 * mu * E + y * trE * np.eye(3)
        return s


class OrthotropicLinearElastic(Material, D3):
    """Orthotropic elastic material definition.

    Matches FEOrthoElastic material in FEBio.  This material is *not* linear.

    """

    # Note: I don't recognize the constitutive model FEBio uses, so it's unclear if
    # this material is "correct" in a broader sense.

    # Permissible values are interdependent, but these are the ranges of potential
    # values for each parameter.
    bounds = {
        "E1": (0, inf),
        "E2": (0, inf),
        "E3": (0, inf),
        "G12": (0, inf),
        "G23": (0, inf),
        "G31": (0, inf),
        "ν12": (-inf, inf),  # min might be -1
        "ν23": (-inf, inf),  # min might be -1
        "ν31": (-inf, inf),  # min might be -1
    }

    def __init__(self, props):
        if len(props) != len(self.bounds):
            raise ValueError(
                f"{len(props)} parameters provided; expected {len(self.bounds)}"
            )
        # Define material properties
        self.E1 = props["E1"]
        self.E2 = props["E2"]
        self.E3 = props["E3"]
        self.G12 = props["G12"]
        self.G23 = props["G23"]
        self.G31 = props["G31"]
        self.ν12 = props["ν12"]
        self.ν23 = props["ν23"]
        self.ν31 = props["ν31"]

        # Verify parameter values
        self.check_parameters_bounds()
        S = orthotropic_elastic_compliance_matrix(self)
        if not is_positive_definite(S):
            raise InvalidParameterError("Stiffness matrix is not positive definite.")

        # Derived properties
        self.ν21 = self.ν12 * self.E2 / self.E1
        self.ν13 = self.ν31 * self.E1 / self.E3
        self.ν32 = self.ν23 * self.E3 / self.E2
        # Derived Lamé
        μ1 = self.G12 + self.G31 - self.G23
        μ2 = self.G12 - self.G31 + self.G23
        μ3 = -self.G12 + self.G31 + self.G23
        self.μ = [μ1, μ2, μ3]
        # Lamé "coefficients"; for compliance matrix
        self.Sλ = np.array(
            [
                [1 / self.E1, -self.ν12 / self.E1, -self.ν31 / self.E3],
                [-self.ν12 / self.E1, 1 / self.E2, -self.ν23 / self.E2],
                [-self.ν31 / self.E3, -self.ν23 / self.E2, 1 / self.E3],
            ]
        )
        # Lamé "constants"; for stiffness matrix
        self.Cλ = np.linalg.inv(self.Sλ) - 2 * np.diag(self.μ)

    @classmethod
    def from_feb(cls, E1, E2, E3, G12, G23, G31, v12, v23, v31, **kwargs):
        # parsing FEBio XML may call from_feb with extra kwargs
        return cls(
            {
                "E1": E1,
                "E2": E2,
                "E3": E3,
                "G12": G12,
                "G23": G23,
                "G31": G31,
                "ν12": v12,
                "ν23": v23,
                "ν31": v31,
            }
        )

    def tstress(self, F, **kwargs):
        """Cauchy stress tensor"""
        C = F.T @ F
        B = F @ F.T
        Q = np.eye(3)  # material orientation is handled by OrientedMaterial
        K = [Q[:, i] @ C @ Q[:, i] for i in range(3)]
        # ^ squared stretch in symmetry plane direction
        a = [F @ Q[:, i] / np.sqrt(K[i]) for i in range(3)]
        A = [np.outer(ai, ai) for ai in a]
        σ = np.zeros((3, 3))
        for i in range(3):
            σ += self.μ[i] * K[i] * (A[i] @ (B - np.eye(3)) + (B - np.eye(3)) @ A[i])
            for j in range(3):
                σ += (
                    0.5
                    * self.Cλ[i, j]
                    * ((K[i] - 1) * K[j] * A[j] + (K[j] - 1) * K[i] * A[i])
                )
        return 0.5 * σ / np.linalg.det(F)


class NeoHookean(Material, D3):
    """Neo-Hookean compressible hyperelastic material.

    Specified in FEBio xml as `neo-Hookean`.

    Note that there are multiple compressible "Neo-Hookean" formulations
    floating around in the literature; this particular one reduces to
    linear elasticity for small strains and small rotations.

    """

    bounds = {
        "E": (0, inf),
        "ν": (-1, 0.5),  # not absolutely sure of this
        "μ": (0, inf),  # not absolutely sure of this
        "λ": (0, inf),  # not absolutely sure of this
    }

    def __init__(self, E, ν):
        y, mu = to_Lamé(E, ν)
        self.mu = mu
        self.y = y

    @classmethod
    def from_feb(cls, E, v):
        """Return instance from FEBio XML-style argument names"""
        return cls(E, v)

    def w(self, F):
        y = self.y
        mu = self.mu
        C = np.dot(F.T, F)
        i1 = np.trace(C)
        J = np.linalg.det(F)
        w = mu / 2.0 * (i1 - 1) - mu * log(J) + y / 2.0 * (log(J)) ** 2.0
        return w

    def tstress(self, F, **kwargs):
        """Cauchy stress tensor."""
        y = self.y
        mu = self.mu
        J = det(F)
        B = dot(F, F.T)  # left cauchy-green
        t = mu / J * (B - np.eye(3)) + y / J * log(J) * np.eye(3)
        return t

    def pstress(self, F, **kwargs):
        """1st Piola-Kirchoff stress."""
        s = self.sstress(F)
        p = dot(F, s)
        return p

    def sstress(self, F, **kwargs):
        """2nd Piola-Kirchoff stress."""
        y = self.y
        mu = self.mu
        J = det(F)
        C = dot(F.T, F)
        Cinv = np.linalg.inv(C)
        s = mu * (np.eye(3) - Cinv) + y * log(J) * Cinv
        return s


class HolmesMow(Material, D3):
    """Holmes-Mow coupled hyperelastic material.

    See page 73 of the FEBio theory manual, version 1.8.

    """

    # TODO: Support open vs. closed intervals
    bounds = {
        "E": (0, inf),
        "ν": (-1, 0.5),
        "β": (0, inf),
    }

    def __init__(self, E, ν, β):
        self.E = E
        self.ν = ν
        self.β = β

    @classmethod
    def from_feb(cls, E, v, beta, **kwargs):
        return cls(E=E, ν=v, β=beta)

    def w(self, F):
        y, mu = to_Lamé(self.E, self.ν)
        C = np.dot(F.T, F)
        i1 = np.trace(C)
        i2 = 0.5 * (i1**2.0 - trace(dot(C, C)))
        J = np.linalg.det(F)
        Q = (
            self.β
            / (y + 2.0 * mu)
            * (
                (2.0 * mu - y) * (i1 - 3.0)
                + y * (i2 - 3.0)
                - (y + 2.0 * mu) * log(J**2.0)
            )
        )
        c = (y + 2.0 * mu) / (2.0 * self.β)
        w = 0.5 * c * (exp(Q) - 1.0)
        return w

    def tstress(self, F, **kwargs):
        """Return Cauchy stress tensor"""
        y, mu = to_Lamé(self.E, self.ν)
        J = det(F)
        B = dot(F, F.T)  # left cauchy-green
        i1 = np.trace(B)
        i2 = 0.5 * (i1**2.0 - trace(dot(B, B)))
        Q = (
            self.β
            / (y + 2.0 * mu)
            * (
                (2.0 * mu - y) * (i1 - 3.0)
                + y * (i2 - 3.0)
                - (y + 2.0 * mu) * log(J**2.0)
            )
        )
        t = (
            1.0
            / (2.0 * J)
            * exp(Q)
            * (
                (2.0 * mu + y * (i1 - 1.0)) * B
                - y * dot(B, B)
                - (y + 2.0 * mu) * eye(3)
            )
        )
        return t

    def pstress(self, F, **kwargs):
        """1st Piola-Kirchoff stress."""
        t = self.tstress(F)
        p = det(F) * dot(t, np.linalg.inv(F).T)
        return p

    def sstress(self, F, **kwargs):
        """2nd Piola-Kirchoff stress."""
        t = self.tstress(F)
        s = det(F) * dot(np.linalg.inv(F), dot(t, np.linalg.inv(F).T))
        return s


class FungOrthotropicElastic(Material, D3):
    """Fung orthotropic elastic model"""

    # TODO: support open vs. closed intervals
    bounds = {
        "E1": (0, inf),
        "E2": (0, inf),
        "E3": (0, inf),
        "G12": (0, inf),
        "G23": (0, inf),
        "G31": (0, inf),
        "ν12": (-inf, inf),  # min might be -1
        "ν23": (-inf, inf),  # min might be -1
        "ν31": (-inf, inf),  # min might be -1
        "c": (0, inf),  # open
        "K": (0, inf),  # open
    }

    def __init__(
        self,
        E1,
        E2,
        E3,
        G12,
        G23,
        G31,
        ν12,
        ν23,
        ν31,
        c,
        K,
    ):
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.G12 = G12
        self.G23 = G23
        self.G31 = G31
        self.ν12 = ν12
        self.ν23 = ν23
        self.ν31 = ν31
        self.c = c
        self.K = K

        # Verify parameter values
        self.check_parameters_bounds()
        S = orthotropic_elastic_compliance_matrix(self)
        if not is_positive_definite(S):
            raise InvalidParameterError("Stiffness matrix is not positive definite.")

        # Lamé parameters
        self.μ = np.array(
            [
                self.G12 + self.G31 - self.G23,
                self.G12 - self.G31 + self.G23,
                -self.G12 + self.G31 + self.G23,
            ]
        )
        uli = np.linalg.inv(
            np.array(
                [
                    [1 / E1, -ν12 / E1, -ν31 / E3],
                    [-ν12 / E1, 1 / E2, -ν23 / E2],
                    [-ν31 / E3, -ν23 / E2, 1 / E3],
                ]
            )
        )
        self.λ = uli - np.diag(2 * self.μ)
        assert np.allclose(self.λ, self.λ.T)

    def tstress(self, F, **kwargs):
        """Return Cauchy stress tensor"""
        J = np.linalg.det(F)
        C = F @ F.T
        E = 0.5 * (C - np.eye(3))
        E2 = E @ E

        q = self.c**-1 * (
            sum(
                [
                    2 * self.μ[a] * E2[a, a]
                    + sum([self.λ[a, b] * E[a, a] * E[b, b] for b in range(3)])
                    for a in range(3)
                ]
            )
        )

        def ddC_q(i, j):
            return self.c**-1 * (
                2 * (self.μ[i] + self.μ[j]) * E[i, j]
                + (i == j) * sum([self.λ[i, b] * E[b, b] for b in range(3)])
                + (i == j) * sum([self.λ[a, j] * E[a, a] for a in range(3)])
            )

        s_mat = (
            0.5
            * self.c
            * np.exp(q)
            * np.array(
                [
                    [ddC_q(0, 0), ddC_q(0, 1), ddC_q(0, 2)],
                    [ddC_q(1, 0), ddC_q(1, 1), ddC_q(1, 2)],
                    [ddC_q(2, 0), ddC_q(2, 1), ddC_q(2, 2)],
                ]
            )
        )
        σ_mat = 1 / J * F @ s_mat @ F.T
        # TODO: parametrize bulk compression law; FEBio supports 3 laws
        σ_bulk = 1 / J * self.K * np.log(J) * np.eye(3)  # ∂U/∂J
        return σ_mat + σ_bulk


class DeviatoricMooneyRivlin(Material, Uncoupled, D3):
    """Deviatoric Mooney–Rivlin material

    Matches deviatoric part of "Mooney-Rivlin" material in FEBio.

    """

    # TODO: support open vs. closed intervals
    bounds = {
        "c1": (0, inf),
        "c2": (0, inf),
    }

    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def tilde_stress(self, F, **kwargs):
        """Return σ_tilde stress

        Cauchy stress σ = dev(σ_tilde).  σ_tilde gets a separate function to support
        use of the material both alone and in a deviatoric mixture.

        """
        J = np.linalg.det(F)
        F_tilde = np.linalg.det(F) ** (-1 / 3) * F
        # everything defined after here is deviatoric
        B = F_tilde @ F_tilde.T
        I1 = np.trace(B)
        # TODO: verify ∂W/∂C
        σ = (
            2 / J * (self.c1 * B + self.c2 * I1 * B - self.c2 * B @ B)
        )  # J is still real J
        return σ


class DeviatoricHGOMatrix(Material, Uncoupled, D3):
    """Deviatoric part of uncoupled Holzapfel–Gasser–Ogden material

    Matches deviatoric matrix part of "Holzapfel-Gasser-Ogden" material in FEBio.

    """

    # TODO: support open vs. closed intervals
    bounds = {
        "μ": (0, inf),
    }

    def __init__(self, μ):
        self.mu = μ

    def tilde_stress(self, F, **kwargs):
        """Cauchy stress tensor"""
        J = np.linalg.det(F)
        # everything defined after here is deviatoric
        F = np.linalg.det(F) ** (-1 / 3) * F
        C = F.T @ F
        I1 = np.trace(C)
        μ = self.mu
        # w = 0.5 * μ * (I1 - 3)
        # t = 2/J F ∂W/∂C F'
        # ∂W/∂C = μ/2 I
        σ = 1 / J * μ * F @ F.T  # J is still real J
        return σ


class DeviatoricHGOFiber3D(Material, Uncoupled, D3):
    """3D formulation of an uncoupled Holzapfel–Gasser–Ogden fiber family

    Matches one fiber part (one summand) of "Holzapfel-Gasser-Ogden" material in FEBio.

    """

    # TODO: support open vs. closed intervals
    bounds = {
        "ξ": (0, inf),  # open
        "α": (0, inf),  # open
        "κ": (0, 1 / 3),  # closed
    }

    def __init__(self, ξ, α, κ):
        self.modulus = ξ
        self.exp_coef = α
        self.dispersion = κ

    def tilde_stress(self, F, **kwargs):
        """Return Cauchy stress tensor"""
        ξ = self.modulus
        α = self.exp_coef
        κ = self.dispersion
        J = np.linalg.det(F)
        # everything defined after here is deviatoric
        F = np.linalg.det(F) ** (-1 / 3) * F
        C = F.T @ F
        I1 = np.trace(C)
        I4 = C[0, 0]  # e1 · C · e1
        Ea = κ * (I1 - 3) + (1 - 3 * κ) * (I4 - 1)
        Ea = Ea * (Ea > 0)  # 〈Ea〉
        ddC_I1 = np.eye(3)
        ddC_I4 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        ddC_W = ξ * Ea * (κ * ddC_I1 + (1 - 3 * κ) * ddC_I4) * np.exp(α * Ea**2)
        σ = 2 / J * F @ ddC_W @ F.T  # J is still real J
        return σ


class VolumetricLinear(Material, D3):
    """Linear hydrostatic (bulk modulus) component"""

    # 1D (pressure) material like DonnanSwelling, but elastic (strain → zero strain)

    bounds = {
        "K": (0, inf),  # open
    }

    def __init__(self, K):
        self.K = K

    def tstress(self, F, **kwargs):
        """Return Cauchy stress"""
        # σ = ∂⁄∂J U(J)
        J = np.linalg.det(F)
        p = self.K * (J - 1)
        return p * np.eye(3)


class VolumetricLogInverse(Material, D3):
    """ln(J)/J hydrostatic (bulk modulus) component"""

    # 1D (pressure) material like DonnanSwelling, but elastic (strain → zero strain)

    bounds = {
        "K": (0, inf),  # open
    }

    def __init__(self, K):
        self.K = K

    def tstress(self, F, **kwargs):
        """Return Cauchy stress"""
        # σ = ∂⁄∂J U(J)
        J = np.linalg.det(F)
        p = self.K * np.log(J) / J
        return p * np.eye(3)


class VolumetricHGO(Material, D3):
    """J - J^-1 hydrostatic (bulk modulus) component

    Note that the 1/2 factor that FEBio uses is omitted here.

    """

    # 1D (pressure) material like DonnanSwelling, but elastic (strain → zero strain)

    bounds = {
        "K": (0, inf),  # open
    }

    def __init__(self, K):
        self.K = K

    def tstress(self, F, **kwargs):
        """Return Cauchy stress"""
        # σ = ∂⁄∂J U(J)
        J = np.linalg.det(F)
        p = self.K * (J - J**-1)
        return p * np.eye(3)


def add_density(material, density=0):
    """Add a density attribute to a material

    This is implemented as a function rather than as a base class because it is
    unclear how to handle densities in a mixture material.  Inheritance would give
    each constituent an individual density, but really only the mixture has a
    density.  Solving this properly may require metaclasses.  This function is a
    stopgap measure.

    """
    if density < 0:
        raise InvalidParameterError(f"Density must be ≥ 0.  Got `{density}`.")
    material.density = density
    return material

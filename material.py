import numpy as np
from numpy import dot, trace, eye, outer
from numpy.linalg import det
from math import log, exp, sin, cos, radians, pi

# Same-package modules
from .core import Sequence, ScaledSequence

_DEFAULT_ORIENT_RANK1 = np.array([1, 0, 0])

_DEFAULT_ORIENT_RANK2 = (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))


def to_Lamé(E, v):
    """Convert Young's modulus & Poisson ratio to Lamé parameters."""
    y = v * E / ((1.0 + v) * (1.0 - 2.0 * v))
    mu = E / (2.0 * (1.0 + v))
    return y, mu


def from_Lamé(y, u):
    """Convert Lamé parameters to modulus & Poisson's ratio."""
    E = u / (y + u) * (2.0 * u + 3.0 * y)
    v = 0.5 * y / (y + u)
    return E, v


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


class OrientedMaterial:
    """A material with an orientation matrix.

    TODO: Needs unit tests.

    """

    def __init__(self, material, Q=np.eye(3)):
        self.material = material
        self.orientation = Q

    def w(self, F):
        return self.material.w(F)

    def tstress(self, F):
        Q = self.orientation
        J = np.linalg.det(F)
        if Q.ndim == 1:
            # 1D material ("fiber")
            λ = np.linalg.norm(F @ Q)  # np.sqrt(Q @ F.T @ F @ Q.T)
            N = Q
            σ_loc_PK2 = self.material.stress(λ) * np.outer(N, N)
            σ_loc = 1 / J * F @ σ_loc_PK2 @ F.T
        elif Q.ndim == 2:
            # 3D material ("solid")
            σ_loc = self.material.tstress(F @ Q)
            # ^ Stress in own local basis.  This is a change of
            # coordinate system for the material, not an observer
            # change, such that material anisotropy is accounted for.
        return σ_loc

    def pstress(self, F):
        Q = self.orientation
        return Q @ self.material.pstress(Q.T @ F)  # TODO: Check

    def sstress(self, F):
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
        return s_loc


class Permeability:
    """Parent type for Permeability implementations."""

    def __init__(self, **kwargs):
        msg = """The Permeability class is meant only to serve as a supertype for
child classes that implement the necessary functionality.  Only its
child classes should be instantiated as objects."""
        raise NotImplementedError(msg)


class IsotropicConstantPermeability(Permeability):
    """Isotropic strain-independent permeability"""

    def __init__(self, k, **kwargs):
        self.k = k

    @classmethod
    def from_feb(cls, perm, **kwargs):
        return cls(perm)


class IsotropicHolmesMowPermeability(Permeability):
    """Isotropic Holmes-Mow permeability"""

    def __init__(self, k0, M, α, **kwargs):
        self.k0 = k0
        self.M = M
        self.α = α

    @classmethod
    def from_feb(cls, perm, M, alpha, **kwargs):
        return cls(perm, M, alpha)


class PoroelasticSolid:
    """Fluid-saturated solid."""

    def __init__(self, solid, permeability: Permeability, solid_fraction):
        """Return PoroelasticSolid instance

        solid := Solid material instance.

        permeability := Permeability instance.

        solid_fraction := Volume fraction of solid.  Volume fraciton of
        solid + volume fraction of fluid = 1.

        """
        self.solid_material = solid
        self.solid_fraction = solid_fraction
        self.permeability = permeability


class DonnanSwelling:
    """Swelling pressure of the Donnan equilibrium type."""

    def __init__(self, phi0_w, fcd0, ext_osm, osm_coef, **kwargs):
        # Bounds checks
        if _is_fixed_property(phi0_w) and not (0 <= phi0_w <= 1):
            raise ValueError(f"phi0_w = {phi0_w}; it is required that 0 ≤ phi0_w ≤ 1")
        if _is_fixed_property(fcd0) and not (fcd0 >= 0):
            raise ValueError(f"fcd0 = {fcd0}; it is required that 0 < fcd0")
        if _is_fixed_property(ext_osm) and not (ext_osm >= 0):
            raise ValueError(f"ext_osm = {ext_osm}; it is required that 0 < ext_osm")
        # Store values
        self.phi0_w = phi0_w
        self.fcd0 = fcd0
        self.ext_osm = ext_osm
        self.osm_coef = osm_coef

    @classmethod
    def from_feb(cls, phiw0, cF0, bosm, Phi=1, **kwargs):
        return cls(phiw0, cF0, bosm, Phi)


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


class SolidMixture:
    """Mixture of solids with no interdependencies or residual stress.

    The strain energy of the mixture is defined as the sum of the
    strain energies for each component.

    """

    def __init__(self, solids, **kwargs):
        """Mixture of solids.

        Parameters
        ----------
        args : object
            Each parameter passed to SolidMixture() should be a
            material object, with functions for strain energy and
            stresses.  The object must also have material properties
            defined.

        """
        self.materials = []
        for solid in solids:
            self.materials.append(solid)

    def w(self, F):
        return sum(material.w(F) for material in self.materials)

    def tstress(self, F):
        return sum(material.tstress(F) for material in self.materials)

    def pstress(self, F):
        return sum(material.pstress(F) for material in self.materials)

    def sstress(self, F):
        return sum(material.sstress(F) for material in self.materials)


class RigidBody:
    """Rigid body."""

    def __init__(self, props={}, **kwargs):
        self.density = 1
        if "density" in props:
            self.density = props["density"]


class ExponentialFiber:
    """1D fiber with exponential power law.

    Coupled formulation ("fiber-exp-pow" in FEBio").

    References
    ----------
    FEBio users manual 2.9, page 144.

    """

    def __init__(self, props):
        self.α = props["alpha"]
        self.β = props["beta"]
        self.ξ = props["ksi"]

    def w(self, λ):
        """Return pseudo-strain energy density."""
        # TODO: Unit test
        α = self.α
        β = self.β
        ξ = self.ξ
        Ψ = ξ / (α * β) * (exp(α * (λ ** 2 - 1) ** β) - 1)
        return Ψ

    def stress(self, λ):
        """Return material stress scalar."""
        α = self.α
        β = self.β
        ξ = self.ξ
        dΨ_dλsq = ξ * (λ ** 2 - 1) ** (β - 1) * exp(α * (λ ** 2 - 1) ** β)
        # Use dΨ_dλsq instead of dΨ_dλ because this is the equivalent of
        # dΨ/dC.
        σ = 2 * dΨ_dλsq * unit_step(λ - 1)
        return σ

    def sstress(self, λ):
        """Synonym for material stress.

        Makes some use cases easier by providing a consistent name with
        3D materials.

        """
        return self.stress(λ)


class ExponentialFiber3D:
    """Fiber with exponential power law.

    Coupled formulation ("fiber-exp-pow" in FEBio").

    This is the deprecated 3D implementation that mixes material
    orientation with the fiber's constitutive law.  Prefer
    ExponentialFiber, which does not mix concerns in this manner.

    References
    ----------
    FEBio users manual 2.0, page 104.

    """

    def __init__(self, props, orientation=_DEFAULT_ORIENT_RANK1, **kwargs):
        self.alpha = props["alpha"]
        self.beta = props["beta"]
        self.xi = props["ksi"]
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
        a = self.alpha
        b = self.beta
        xi = self.xi
        w = xi / (a * b) * (exp(a * (In - 1.0) ** b) - 1.0)
        return w

    def tstress(self, F):
        """Return Cauchy stress tensor aligned to local axes."""
        F = np.array(F)
        # Components
        J = det(F)
        C = dot(F.T, F)
        # fiber unit vector
        N = self.orientation
        # Square of fiber stretch
        In = dot(N, dot(C, N))
        yf = In ** 0.5  # fiber stretch
        n = dot(F, N) / yf

        a = self.alpha
        b = self.beta
        xi = self.xi
        dPsi_dIn = xi * (In - 1.0) ** (b - 1.0) * exp(a * (In - 1.0) ** b)
        t = (2 / J) * unit_step(In - 1.0) * In * dPsi_dIn * outer(n, n)
        return t

    def pstress(self, F):
        """Return 1st Piola–Kirchoff stress tensor in local axes."""
        t = self.tstress(F)
        p = det(F) * dot(t, np.linalg.inv(F).T)
        return p

    def sstress(self, F):
        """Return 2nd Piola-Kirchoff stress tensor in local axes."""
        t = self.tstress(F)
        s = det(F) * dot(np.linalg.inv(F), dot(t, np.linalg.inv(F).T))
        return s


class PowerLinearFiber:
    """1D fiber with power–linear law.

    Equivalent to "fiber-pow-lin" or "fiber-power-lin" in FEBio, treated
    as a 1D material.

    References
    ----------
    FEBio users manual 2.9, page 173 (not page 146; the manual shows the
    wrong equations for the "fiber-pow-lin" entry).

    """

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
        """Return material stress scalar."""
        λ0 = self.λ0
        β = self.β
        E = self.E
        ξ = E / 2 / (β - 1) * λ0 ** (-3) * (λ0 ** 2 - 1) ** (2 - β)
        b = E / 2 / (λ0 ** 3) * ((λ0 ** 2 - 1) / (2 * (β - 1)) + λ0 ** 2)
        # Stress
        λsq = λ ** 2
        if λ <= 1:
            σ = 0
        elif λ <= λ0:
            dΨ_dλsq = ξ / 2 * (λsq - 1) ** (β - 1)
            σ = 2 * dΨ_dλsq  # equiv to 2 dΨ/dC; PK2
        else:
            dΨ_dλsq = b - E / 2 / λ
            σ = 2 * dΨ_dλsq  # equiv to 2 dΨ/dC; PK2
        return σ

    def sstress(self, λ):
        """Synonym for material stress.

        Makes some use cases easier by providing a consistent name with
        3D materials.

        """
        return self.stress(λ)


class PowerLinearFiber3D:
    """Fiber with piecewise power-law (toe) and linear regions.

    Coupled formulation ("fiber-pow-lin" or "fiber-power-lin" in FEBio).

    This is the deprecated 3D implementation that mixes material
    orientation with the fiber's constitutive law.  Prefer
    ExponentialFiber, which does not mix concerns in this manner.

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

    def tstress(self, F):
        """Return Cauchy stress tensor aligned to local axes."""
        # Properties
        I0 = self.λ0 ** 2.0
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

    def pstress(self, F):
        """Return 1st Piola–Kirchoff stress tensor aligned to local axes."""
        raise NotImplementedError

    def sstress(self, F):
        """Return 2nd Piola–Kirchoff stress tensor aligned to local axes."""
        raise NotImplementedError


class HolmesMow:
    """Holmes-Mow coupled hyperelastic material.

    See page 73 of the FEBio theory manual, version 1.8.

    """

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
        i2 = 0.5 * (i1 ** 2.0 - trace(dot(C, C)))
        J = np.linalg.det(F)
        Q = (
            self.β
            / (y + 2.0 * mu)
            * (
                (2.0 * mu - y) * (i1 - 3.0)
                + y * (i2 - 3.0)
                - (y + 2.0 * mu) * log(J ** 2.0)
            )
        )
        c = (y + 2.0 * mu) / (2.0 * self.β)
        w = 0.5 * c * (exp(Q) - 1.0)
        return w

    def tstress(self, F):
        """Cauchy stress tensor."""
        y, mu = to_Lamé(self.E, self.ν)
        J = det(F)
        B = dot(F, F.T)  # left cauchy-green
        i1 = np.trace(B)
        i2 = 0.5 * (i1 ** 2.0 - trace(dot(B, B)))
        Q = (
            self.β
            / (y + 2.0 * mu)
            * (
                (2.0 * mu - y) * (i1 - 3.0)
                + y * (i2 - 3.0)
                - (y + 2.0 * mu) * log(J ** 2.0)
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

    def pstress(self, F):
        """1st Piola-Kirchoff stress."""
        t = self.tstress(F)
        p = det(F) * dot(t, np.linalg.inv(F).T)
        return p

    def sstress(self, F):
        """2nd Piola-Kirchoff stress."""
        t = self.tstress(F)
        s = det(F) * dot(np.linalg.inv(F), dot(t, np.linalg.inv(F).T))
        return s


class IsotropicElastic:
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
        W = 0.5 * y * trE ** 2.0 + mu * np.sum(E * E)
        return W

    def tstress(self, F):
        """Cauchy stress."""
        s = self.sstress(F)
        J = np.linalg.det(F)
        t = np.dot(np.dot(F, s), F.T) / J
        return t

    def pstress(self, F):
        """1st Piola-Kirchoff stress."""
        s = self.sstress(F)
        p = np.dot(s, F.T)
        return p

    def sstress(self, F):
        """2nd Piola-Kirchoff stress."""
        y = self.y
        mu = self.mu
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        s = y * trE * np.eye(3) + 2.0 * mu * E
        return s


class OrthotropicElastic:
    """Orthotropic elastic material definition.

    Matches FEOrthoElastic material in FEBio.  This material is *not*
    linear.

    """

    # Note: I don't recognize the constitutitive model FEBio uses, so
    # it's unclear if this material is "correct" in a broader sense.

    def __init__(self, props):
        # Define material properties
        self.E1 = props["E1"]
        self.E2 = props["E2"]
        self.E3 = props["E3"]
        self.G12 = props["G12"]
        self.G23 = props["G23"]
        self.G31 = props["G31"]
        self.v12 = props["ν12"]
        self.v23 = props["ν23"]
        self.v31 = props["ν31"]
        # Derived properties
        self.v21 = self.v12 * self.E2 / self.E1
        self.v13 = self.v31 * self.E1 / self.E3
        self.v32 = self.v23 * self.E3 / self.E2
        # Derived Lamé
        μ1 = self.G12 + self.G31 - self.G23
        μ2 = self.G12 - self.G31 + self.G23
        μ3 = -self.G12 + self.G31 + self.G23
        self.μ = [μ1, μ2, μ3]
        # Lamé "coefficients"; for compliance matrix
        self.Sλ = np.array(
            [
                [1 / self.E1, -self.v12 / self.E1, -self.v31 / self.E3],
                [-self.v12 / self.E1, 1 / self.E2, -self.v23 / self.E2],
                [-self.v31 / self.E3, -self.v23 / self.E2, 1 / self.E3],
            ]
        )
        # Lamé "constants"; for stiffness matrix
        self.Cλ = np.linalg.inv(self.Sλ) - 2 * np.diag(self.μ)

    @classmethod
    def from_feb(cls, E1, E2, E3, G12, G23, G31, v12, v23, v31):
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

    def tstress(self, F):
        """Cauchy stress tensor."""
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


class NeoHookean:
    """Neo-Hookean compressible hyperelastic material.

    Specified in FEBio xml as `neo-Hookean`.

    Note that there are multiple compressible "Neo-Hookean" formulations
    floating around in the literature; this particular one reduces to
    linear elasticity for small strains and small rotations.

    """

    def __init__(self, props, **kwargs):
        if "E" in props and "v" in props:
            y, mu = to_Lamé(props["E"], props["v"])
        elif "lambda" in props and "mu" in props:
            y = props["lambda"]
            mu = props["lambda"]
        else:
            raise Exception(
                "The combination of material properties "
                "in " + str(props) + " is not yet "
                "implemented."
            )
        self.mu = mu
        self.y = y

    def w(self, F):
        y = self.y
        mu = self.mu
        C = np.dot(F.T, F)
        i1 = np.trace(C)
        J = np.linalg.det(F)
        w = mu / 2.0 * (i1 - 1) - mu * log(J) + y / 2.0 * (log(J)) ** 2.0
        return w

    def tstress(self, F):
        """Cauchy stress tensor."""
        y = self.y
        mu = self.mu
        J = det(F)
        B = dot(F, F.T)  # left cauchy-green
        t = mu / J * (B - np.eye(3)) + y / J * log(J) * np.eye(3)
        return t

    def pstress(self, F):
        """1st Piola-Kirchoff stress."""
        s = self.sstress(F)
        p = dot(F, s)
        return p

    def sstress(self, F):
        """2nd Piola-Kirchoff stress."""
        y = self.y
        mu = self.mu
        J = det(F)
        C = dot(F.T, F)
        Cinv = np.linalg.inv(C)
        s = mu * (np.eye(3) - Cinv) + y * log(J) * Cinv
        return s

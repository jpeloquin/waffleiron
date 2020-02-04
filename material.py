import numpy as np
from numpy import dot, trace, eye, outer
from numpy.linalg import det
from math import log, exp, sin, cos, radians, pi
import febtools as feb
# Same-package modules
from .core import Sequence, ScaledSequence


def tolame(E, v):
    """Convert Young's modulus & Poisson ratio to Lamé parameters.

    """
    y = v * E / ((1.0 + v) * (1.0 - 2.0 * v))
    mu = E / (2.0 * (1.0 + v))
    return y, mu


def fromlame(y, u):
    """Convert Lamé parameters to modulus & Poisson's ratio.

    """
    E = u / (y + u) * (2.0 * u + 3.0 * y)
    v = 0.5 * y / (y + u)
    return E, v


def _is_fixed_property(p):
    if isinstance(p, Sequence) or isinstance(p, ScaledSequence):
        return False
    else:
        return True


class Permeability:
    """Parent type for Permeability implementations."""

    def __init__(self):
        msg = """The Permeability class is meant only to serve as a supertype for
child classes that implement the necessary functionality.  Only its
child classes should be instantiated as objects."""
        raise NotImplementedError(msg)


class IsotropicConstantPermeability(Permeability):
    """Isotropic strain-independent permeability"""

    def __init__(self, k):
        self.k = k

    @classmethod
    def from_feb(cls, perm, **kwargs):
        return cls(perm)


class IsotropicHolmesMowPermeability(Permeability):
    """Isotropic Holmes-Mow permeability"""
    def __init__(self, k0, M, α):
        self.k0 = k0
        self.M = M
        self.α = α

    @classmethod
    def from_feb(cls, perm, M, alpha, **kwargs):
        return cls(perm, M, alpha)


class PoroelasticSolid:
    """Fluid-saturated solid.

    Currently only isotropic permeability is allowed.

    """
    def __init__(self, solid, permeability: Permeability):
        self.solid_material = solid
        self.permeability = permeability


class DonnanSwelling:
    """Swelling pressure of the Donnan equilibrium type."""
    def __init__(self, phi0_w, fcd0, ext_osm, osm_coef):
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
    def __init__(self, generations):
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

    def __init__(self, solids):
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
        return sum(material.w(F)
                   for material in self.materials)

    def tstress(self, F):
        return sum(material.tstress(F)
                   for material in self.materials)

    def pstress(self, F):
        return sum(material.pstress(F)
                   for material in self.materials)

    def sstress(self, F):
        return sum(material.sstress(F)
                   for material in self.materials)


class RigidBody:
    """Rigid body.

    """
    def __init__(self, props={}):
        self.density = 1
        if "density" in props:
            self.density = props["density"]


class ExponentialFiber:
    """Fiber with exponential power law.

    This is the coupled formulation ("fiber-exp-pow" in FEBio").

    References
    ----------
    FEBio users manual 2.0, page 104.

    """
    def __init__(self, props):
        self.alpha = props['alpha']
        self.beta = props['beta']
        self.xi = props['ksi']
        self.theta = radians(props['theta'])
        # ^ azimuth; 0° := +x, 90° := +y
        self.phi = radians(props['phi'])
        # ^ zenith; 0° := +z, 90° := x-y plane

    def w(self, F):
        """Strain energy density.

        Input angles in degrees.

        """
        F = np.array(F)

        # deviatoric components
        # J = det(F)
        # Fdev = J**(-1.0/3.0) * F
        # Cdev = dot(Fdev.T, Fdev)
        C = dot(F.T, F)
        # fiber unit vector
        N = np.array([sin(self.phi) * cos(self.theta),
                      sin(self.phi) * sin(self.theta),
                      cos(self.phi)])
        # square of fiber stretch
        In = dot(N, dot(C, N))
        a = self.alpha
        b = self.beta
        xi = self.xi
        w = xi / (a * b) * (exp(a * (In - 1.0)**b) - 1.0)
        return w

    def tstress(self, F):
        """Cauchy stress tensor.

        """
        F = np.array(F)

        def H(x):
            """Unit step function."""
            if x > 0.0:
                return 1.0
            else:
                return 0.0

        # Components
        J = det(F)
        C = dot(F.T, F)
        # fiber unit vector
        N = np.array([sin(self.phi) * cos(self.theta),
                      sin(self.phi) * sin(self.theta),
                      cos(self.phi)])
        # Square of fiber stretch
        In = dot(N, dot(C, N))
        yf = In**0.5  # fiber stretch
        n = dot(F, N) / yf

        a = self.alpha
        b = self.beta
        xi = self.xi
        dPsi_dIn = xi * (In - 1.0)**(b - 1.0) * exp(a * (In - 1.0)**b)
        t = (2 / J) * H(In - 1.0) * In * dPsi_dIn * outer(n, n)
        return t

    def pstress(self, F):
        """1st Piola-Kirchoff stress.

        """
        t = self.tstress(F)
        p = det(F) * dot(t, np.linalg.inv(F).T)
        return p

    def sstress(self, F):
        """2nd Piola-Kirchoff stress.

        """
        t = self.tstress(F)
        s = det(F) * dot(np.linalg.inv(F),
                         dot(t, np.linalg.inv(F).T))
        return s


class PowerLinearFiber:
    """Fiber with piecewise power-law (toe) and linear regions.

    Same as "fiber-pow-lin" in FEBio XML.

    """
    def __init__(self, E, β, λ0, azimuth=0, zenith=pi/2):
        self.E = E  # fiber modulus in linear region
        self.β = β  # power law exponent in power law region
        self.λ0 = λ0  # stretch ratio at which power law region
                      # transitions to linear region
        self.azimuth = azimuth
        self.zenith = zenith
        # TODO: Harmonize representation of fiber angle between
        # ExponentialFiber and PowerLinearFiber.  Use `azimuth` and
        # `zenith` everywhere.

    @classmethod
    def from_feb(cls, E, beta, lam0, theta, phi):
        return cls(E, beta, lam0, azimuth=radians(theta), zenith=radians(phi))


class HolmesMow:
    """Holmes-Mow coupled hyperelastic material.

    See page 73 of the FEBio theory manual, version 1.8.

    """
    beta = 0
    y = 0
    mu = 0

    def __init__(self, props):
        if 'E' in props and 'v' in props:
            y, mu = feb.material.tolame(props['E'], props['v'])
        elif 'lambda' in props and 'mu' in props:
            y = props['lambda']
            mu = props['lambda']
        else:
            raise Exception('The combination of material properties '
                            'in ' + str(props) + ' is not yet '
                            'implemented.')
        self.mu = mu
        self.y = y
        self.beta = props['beta']

    def w(self, F):
        b = self.beta
        y = self.y
        mu = self.mu
        C = np.dot(F.T, F)
        i1 = np.trace(C)
        i2 = 0.5 * (i1**2.0 - trace(dot(C, C)))
        J = np.linalg.det(F)
        Q = b / (y + 2.0 * mu) * ((2.0 * mu - y) * (i1 - 3.0)
                                  + y * (i2 - 3.0)
                                  - (y + 2.0*mu) * log(J**2.0))
        c = (y + 2.0 * mu) / (2.0 * b)
        w = 0.5 * c * (exp(Q) - 1.0)
        return w

    def tstress(self, F):
        """Cauchy stress tensor.

        """
        y = self.y
        mu = self.mu
        b = self.beta

        J = det(F)
        B = dot(F, F.T)  # left cauchy-green
        i1 = np.trace(B)
        i2 = 0.5 * (i1**2.0 - trace(dot(B, B)))
        Q = b / (y + 2.0 * mu) * ((2.0 * mu - y) * (i1 - 3.0)
                                  + y * (i2 - 3.0)
                                  - (y + 2.0*mu) * log(J**2.0))
        t = 1.0 / (2.0 * J) * exp(Q) * ((2.0 * mu + y * (i1 - 1.0)) * B
                                        - y * dot(B, B)
                                        - (y + 2.0*mu) * eye(3))
        return t

    def pstress(self, F):
        """1st Piola-Kirchoff stress.

        """
        t = self.tstress(F)
        p = det(F) * dot(t, np.linalg.inv(F).T)
        return p

    def sstress(self, F):
        """2nd Piola-Kirchoff stress.

        """
        t = self.tstress(F)
        s = det(F) * dot(np.linalg.inv(F),
                         dot(t, np.linalg.inv(F).T))
        return s


class IsotropicElastic:
    """Isotropic elastic material definition.

    """
    def __init__(self, props):
        if 'E' in props and 'v' in props:
            y, mu = feb.material.tolame(props['E'], props['v'])
        elif 'lambda' in props and 'mu' in props:
            y = props['lambda']
            mu = props['mu']
        else:
            raise Exception('The combination of material properties '
                            'in ' + str(props) + ' is not yet '
                            'implemented.')
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

    def tstress(self, F):
        """Cauchy stress.

        """
        s = self.sstress(F)
        J = np.linalg.det(F)
        t = np.dot(np.dot(F, s), F.T) / J
        return t

    def pstress(self, F):
        """1st Piola-Kirchoff stress.

        """
        s = self.sstress(F)
        p = np.dot(s, F.T)
        return p

    def sstress(self, F):
        """2nd Piola-Kirchoff stress.

        """
        y = self.y
        mu = self.mu
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        s = y * trE * np.eye(3) + 2.0 * mu * E
        return s


class LinearOrthotropicElastic:
    """Linear orthotropic elastic material definition.

    """
    def __init__(self, props,
                 axes=((1, 0, 0),
                       (0, 1, 0),
                       (0, 0, 1))):
        # Define material properties
        self.E1 = props['E1']
        self.E2 = props['E2']
        self.E3 = props['E3']
        self.G12 = props['G12']
        self.G23 = props['G23']
        self.G31 = props['G31']
        self.v12 = props['ν12']
        self.v23 = props['ν23']
        self.v31 = props['ν31']
        # Define basis vectors
        self.x1 = axes[0]
        self.x2 = axes[1]
        self.x3 = axes[2]


class NeoHookean:
    """Neo-Hookean compressible hyperelastic material.

    Specified in FEBio xml as `neo-Hookean`.

    Note that there are multiple compressible "Neo-Hookean" formulations
    floating around in the literature; this particular one reduces to
    linear elasticity for small strains and small rotations.

    """
    y = None
    mu = None

    def __init__(self, props):
        if 'E' in props and 'v' in props:
            y, mu = feb.material.tolame(props['E'], props['v'])
        elif 'lambda' in props and 'mu' in props:
            y = props['lambda']
            mu = props['lambda']
        else:
            raise Exception('The combination of material properties '
                            'in ' + str(props) + ' is not yet '
                            'implemented.')
        self.mu = mu
        self.y = y

    def w(self, F):
        y = self.y
        mu = self.mu
        C = np.dot(F.T, F)
        i1 = np.trace(C)
        J = np.linalg.det(F)
        w = mu / 2.0 * (i1 - 1) - mu * log(J) + y / 2.0 * (log(J))**2.0
        return w

    def tstress(self, F):
        """Cauchy stress tensor.

        """
        y = self.y
        mu = self.mu
        J = det(F)
        B = dot(F, F.T)  # left cauchy-green
        t = mu / J * (B - np.eye(3)) + y / J * log(J) * np.eye(3)
        return t

    def pstress(self, F):
        """1st Piola-Kirchoff stress.

        """
        s = self.sstress(F)
        p = dot(F, s)
        return p

    def sstress(self, F):
        """2nd Piola-Kirchoff stress.

        """
        y = self.y
        mu = self.mu
        J = det(F)
        C = dot(F.T, F)
        Cinv = np.linalg.inv(C)
        s = mu * (np.eye(3) - Cinv) + y * log(J) * Cinv
        return s

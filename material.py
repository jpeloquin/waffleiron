# coding=utf-8

import numpy as np
from numpy import dot, trace, eye, outer
from numpy.linalg import det
import math
from math import log, exp, sin, cos
import febtools as feb

def getclass(matname):
    """Return reference to a material's class from its name.

    The material names are those used by FEBio.  These names are
    listed in the type attribute of each `material` tag in an `.feb`
    file.

    """
    d = {'isotropic elastic': IsotropicElastic,
         'Holmes-Mow': HolmesMow,
         'fiber-exp-pow': ExponentialFiber,
         'solid mixture': SolidMixture}
    return d[matname]

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
        return sum(material.pstress(F)
                   for material in self.materials)


class ExponentialFiber:
    """Fiber with exponential power law.

    References
    ----------
    FEBio theory manual 1.8, page 78.

    """
    def __init__(self, props):
        self.alpha = props['alpha']
        self.beta = props['beta']
        self.xi = props['ksi']
        self.theta = math.radians(props['theta'])
            # azimuth; 0° := +x, 90° := +y
        self.phi = math.radians(props['phi'])
            # zenith; 0° := +z, 90° := x-y plane

    def w(self, F):
        """(Deviatoric) strain energy density.

        Input angles in degrees.

        """
        F = np.array(F)

        # deviatoric components
        J = det(F)
        Fdev = J**(-1.0/3.0) * F
        Cdev = dot(Fdev.T, Fdev)
        # fiber unit vector
        N = np.array([sin(self.phi) * cos(self.theta),
                      sin(self.phi) * sin(self.theta),
                      cos(self.phi)])
        # square of fiber stretch
        In = dot(N, dot(Cdev, N))
        a = self.alpha
        b = self.beta
        xi = self.xi
        w = xi / (a * b) * (exp(a * (In - 1.0)**b) - 1.0)
        return w

    def tstress(self, F):
        """(Deviatoric) Cauchy stress tensor.

        """
        F = np.array(F)

        def H(x):
            """Unit step function."""
            if x > 0.0:
                return 1.0
            else:
                return 0.0

        # Deviatoric components
        J = det(F)
        Fdev = J**(-1.0/3.0) * F
        Cdev = dot(Fdev.T, Fdev)
        # fiber unit vector
        N = np.array([sin(self.phi) * cos(self.theta),
                      sin(self.phi) * sin(self.theta),
                      cos(self.phi)])
        # (deviatoric) square of fiber stretch
        In = dot(N, dot(Cdev, N))
        n = dot(Fdev, N) / In**0.5

        a = self.alpha
        b = self.beta
        xi = self.xi
        DxIn = xi * (In - 1.0)**(b - 1.0) * exp(a * (In - 1.0)**b)
        t = 2 / J * H(In - 1.0) * In * DxIn * outer(n, n)
        return t

    def pstress(self, F):
        """1st Piola-Kirchoff stress.

        """
        pass


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
        B = dot(F, F.T) # left cauchy-green
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
        t = cls.tstress(F, props)
        p = det(F) * dot(inv(F), t)
        return p


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
        y = self.y
        mu = self.mu
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        s = y * trE * np.eye(3) + 2.0 * mu * E  # 2nd P-K
        J = np.linalg.det(F)
        t = np.dot(np.dot(F, s), F.T) / J
        return t

    def pstress(self, F):
        """1st Piola-Kirchoff stress.

        """
        y = self.y
        mu = self.mu
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        s = y * trE * np.eye(3) + 2.0 * mu * E  # 2nd P-K
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

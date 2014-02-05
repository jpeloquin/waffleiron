# coding=utf-8

import numpy as np
from numpy import dot, trace, eye, outer
from numpy.linalg import det
from math import log, exp, sin, cos

def getclass(matname):
    """Return reference to a material's class from its name.

    The material names are those used by FEBio.  These names are
    listed in the type attribute of each `material` tag in an `.feb`
    file.

    """
    d = {'isotropic elastic': IsotropicElastic,
         'Holmes-Mow': HolmesMow,
         'fiber-exp-pow': ExponentialFiber}
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


class ExponentialFiber:
    """Fiber with exponential power law.

    References
    ----------
    FEBio theory manual 1.8, page 78.

    """
    @staticmethod
    def w(F, props):
        """(Deviatoric) strain energy density.

        Input angles in degrees.

        """
        F = np.array(F)
        xi = props['ksi']
        a = props['alpha']
        b = props['beta']
        theta = props['theta'] # azimuth; 0° := +x, 90° := +y
        phi = props['phi'] # zenith; 0° := +z, 90° := x-y plane
        # deviatoric components
        J = det(F)
        Fdev = J**(-1.0/3.0) * F
        Cdev = dot(Fdev.T, Fdev)
        # fiber unit vector
        N = np.array([sin(phi) * cos(theta),
                      sin(phi) * sin(theta),
                      cos(phi)])
        # square of fiber stretch
        In = dot(N, dot(Cdev, N))
        w = xi / (a * b) * (exp(a * (In - 1.0)**b) - 1.0)
        return w

    @staticmethod
    def tstress(F, props):
        """(Deviatoric) Cauchy stress tensor.

        """
        # unpack properties
        F = np.array(F)
        xi = props['ksi']
        a = props['alpha']
        b = props['beta']
        theta = props['theta'] # azimuth; 0° := +x, 90° := +y
        phi = props['phi'] # zenith; 0° := +z, 90° := x-y plane

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
        N = np.array([sin(phi) * cos(theta),
                      sin(phi) * sin(theta),
                      cos(phi)])
        # (deviatoric) square of fiber stretch
        In = dot(N, dot(Cdev, N))
        n = dot(Fdev, N) / In**0.5

        DxIn = xi * (In - 1.0)**(b - 1.0) * exp(a * (In - 1.0)**b)
        t = 2 / J * H(In - 1.0) * In * DxIn * outer(n, n)
        return t


class HolmesMow:
    """Holmes-Mow coupled hyperelastic material.

    See page 73 of the FEBio theory manual, version 1.8.

    """
    @staticmethod
    def w(F, props):
        b = props['beta']
        y = props['lambda']
        u = props['mu']
        C = np.dot(F.T, F)
        i1 = np.trace(C)
        i2 = 0.5 * (i1**2.0 - trace(dot(C, C)))
        J = np.det(F)
        Q = b / (y + 2.0 * mu) * ((2.0 * mu - y) * (i1 - 3.0)
                                  + y * (i2 - 3.0)
                                  - (y + 2.0*mu) * log(J**2.0))
        c = (y + 2.0 * mu) / (2.0 * b)
        w = 0.5 * c * (exp(Q) - 1.0)
        return w

    @staticmethod
    def tstress(F, props):
        """Cauchy stress tensor.

        """
        y = props['lambda']
        mu = props['mu']
        b = props['beta']

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

    @classmethod
    def pstress(cls, F, props):
        """1st Piola-Kirchoff stress.

        """
        t = cls.tstress(F, props)
        p = det(F) * dot(inv(F), t)
        return p


class IsotropicElastic:
    """Isotropic elastic material definition.

    """

    @staticmethod
    def w(F, props):
        """Strain energy for isotropic elastic material.
        
        F = deformation tensor
        y = Lamé parameter λ
        mu = Lamé parameter μ

        """
        y = props['lambda']
        mu = props['mu']
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        W = 0.5 * y * trE**2.0 + mu * np.sum(E * E)
        return W

    @staticmethod
    def tstress(F, props):
        """Cauchy stress.

        """
        y = props['lambda']
        mu = props['mu']
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        s = y * trE * np.eye(3) + 2.0 * mu * E  # 2nd P-K
        J = np.linalg.det(F)
        t = np.dot(np.dot(F, s), F.T) / J
        return t

    @staticmethod
    def pstress(F, props):
        """1st Piola-Kirchoff stress.

        """
        y = props['lambda']
        mu = props['mu']
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        s = y * trE * np.eye(3) + 2.0 * mu * E  # 2nd P-K
        p = np.dot(s, F.T)
        return p

    @staticmethod
    def sstress(F, props):
        """2nd Piola-Kirchoff stress.

        """
        y = props['lambda']
        mu = props['mu']
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        s = y * trE * np.eye(3) + 2.0 * mu * E
        return s

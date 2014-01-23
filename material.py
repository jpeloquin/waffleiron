# coding=utf-8

import numpy as np

def getclass(matname):
    """Return reference to a material's class from its name.

    The material names are those used by FEBio.  These names are
    listed in the type attribute of each `material` tag in an `.feb`
    file.

    """
    d = {'isotropic elastic': IsotropicElastic}
    return d[matname]

class IsotropicElastic:
    """Isotropic elastic material definition.

    """
    @staticmethod
    def tolame(E, v):
        """Convert Young's modulus & Poisson ratio to Lamé parameters.

        """
        y = v * E / ((1.0 + v) * (1.0 - 2.0 * v))
        mu = E / (2.0 * (1.0 + v))
        return y, mu

    @staticmethod
    def fromlame(y, u):
        """Convert Lamé parameters to modulus & Poisson's ratio.

        """
        E = u / (y + u) * (2.0 * u + 3.0 * y)
        v = 0.5 * y / (y + u)
        return E, v

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

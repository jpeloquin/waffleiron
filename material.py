# coding=utf-8

import numpy as np

class IsotropicElastic:
    """Isotropic elastic material definition.

    """
    @staticmethod
    def lameparam(E, v):
        """Convert Young's modulus & Poisson ratio to Lamé parameters.

        """
        y = v * E / ((1.0 + v) * (1.0 - 2.0 * v))
        mu = E / (2.0 * (1.0 + v))
        return y, mu

    @staticmethod
    def w(F, y, mu):
        """Strain energy for isotropic elastic material.
        
        F = deformation tensor
        y = Lamé parameter λ
        mu = Lamé parameter μ

        """
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        W = 0.5 * y * trE**2.0 + mu * np.sum(E * E)
        return W

    @staticmethod
    def s(F, y, mu):
        """Cauchy stress.

        """
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        trE = np.trace(E)
        # second piola-kirchoff
        s = y * trE * np.eye(3) + 2.0 * mu * E
        # cauchy
        J = np.linalg.det(F)
        t = np.dot(np.dot(F, s), F.T) / J
        return t

import numpy as np

def isotropic_elastic(F, lam, mu):
    """Strain energy for isotropic elastic material.

    F = deformation tensor
    lam = Lamé parameter λ
    mu = Lamé parameter μ

    """
    B = np.dot(F, F.T)
    trE = 0.5 * (np.trace(B) - 3) # from FEBio
    E = 0.5 * (np.dot(F.T, F) - np.eye(3))
    W = 0.5 * lam * trE**2.0 + mu * np.inner(E, E)
    # Cauchy stress
    s = B * (lam * trE - mu) + np.dot(B, B) * mu
    return W, s

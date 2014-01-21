import numpy as np

def jintegral(elements, q):
    """Calculate J integral over elements

    Parameters
    ----------
    elements : list of element objects
       The elements in this list define the domain for evaluation
       of the J integral.

    q : function
       Must be callable as q(x).

    """
    for e in elements:
        sigma = 0
        for p, wp in zip(element.gloc, element.wp):
            t = element.t(p)
            dudx1 = None
            w = element.w(p)
            dqdx = None
            summ = 0
            for i in (1, 2, 3):
                for j in (1, 2, 3):
                    for k in (1, 2, 3):
                        v = (t[i,j] * dudx1[j] - w * (i == 1)) * dqdx[i] * np.det(dxdeta[j, k]) * wp
                        summ = summ + v
                        

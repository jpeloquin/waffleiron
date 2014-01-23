import numpy as np
import febtools

def jintegral(elements, u, q, material_map):
    """Calculate J integral over elements

    Parameters
    ----------
    elements : list of element objects
       The elements in this list define the domain for evaluation
       of the J integral.

    q : function
       Must be callable as q(x).

    """
    def integrand(e, r, u, q, material_map):
        matname = material_map[e.matl_id]['type']
        matl = febtools.material.getclass(matname)
        matlprops = material_map[e.matl_id]['properties']
        F = e.f(r, u)
        p = matl.pstress(F, matlprops) # 1st Piola-Kirchoff stress
        dudx = F - np.eye(3)
        dudx1 = dudx[:,0]
        w = matl.w(F, matlprops) # strain energy
        
        dqdx = e.dinterp(r, q)
        # w * dqdx[0]

        return 1
    j = 0
    for e in elements:
        j += e.integrate(integrand, u, q, material_map)
    return j
                        

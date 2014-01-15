import numpy as np

def jintegral(soln, x):
    """Calculate J integral centered on node

    Parameters
    ----------
    soln : MeshSolution object
        The object (from `febtools`) containing the FEA solution.
    x : 2- or 3- element array-like
        (x, y) or (x, y, z) position of node at notch tip.
    """
    node_id = soln.find_nearest_node(*x)
    area1 = soln.elem_of_node(node_id)
    area2 = soln.conn_elem(area1)
    print(area1)
    print(area2)
    for element in area1:
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
                        

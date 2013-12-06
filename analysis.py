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

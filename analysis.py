import numpy as np
import febtools

def select_elems_around_node(mesh, i, n=3):
    """Select elements centered on node i.

    Parameters
    ----------

    mesh : febtools.mesh.Mesh object

    i : integer
        The index of the central node.

    r : integer, optional
        The number of concentric rings of elements to select.

    """
    nodelist = set([i])
    elements = set([])
    for n in xrange(n):
        for i in nodelist:
            elements = elements | mesh.elem_with_node(i)
        nodelist = set(i for e in elements
                       for i in e.inode)
        # ^ This needlessly re-adds elements already in the domain;
        # there's probably a better way; see undirected graph search.
        # Doing this efficiently requires transforming the mesh into a
        # graph data structure.
    return elements

def jdomain(mesh, inode_tip, n=3, qtype='plateau'):
    """Define q for for the J integral.

    """
    q = [None] * len(mesh.node)
    inner_elements = select_elems_around_node(mesh, inode_tip, n=n-1)
    inner_nodes = set(i for e in inner_elements
                      for i in e.inode)
    elements = select_elems_around_node(mesh, inode_tip, n=n)
    nodes = set(i for e in elements
                for i in e.inode)
    def node_connectivity(elements, n_nodes):
        connectivity = [0] * n_nodes
        for e in elements:
            for i in e.inode:
                connectivity[i] += 1
        return connectivity
    c = node_connectivity(mesh.element, len(mesh.node))
    crack_nodes = [inode_tip]
    # walk along crack boundary to find crack nodes
    for l in xrange(n):
        crack_nodes = set(idx for i in crack_nodes
                          for e in mesh.elem_with_node(i)
                          for idx in e.inode
                          if c[idx] == e.n/2)
    crack_nodes = crack_nodes | set([inode_tip])
    if qtype == 'plateau':
        for i in inner_nodes:
            q[i] = 1.0
        for i in nodes - inner_nodes:
            q[i] = 0.0
        for i in crack_nodes & inner_nodes:
            q[i] = 1.0
    else:
        raise Exception('{}-type q functions are not implemented '
                        'yet.'.format(qtype))
    return elements, q

def jintegral(elements, u, q, material_map):
    """Calculate J integral.

    Parameters
    ----------
    elements : list of element objects
       List the elements to use as the integration domain.

    q_mode : list of scalars
       Specify a list of q values, one per node.

    """

    def integrand(e, r, u, q, material_map):
        matname = material_map[e.matl_id]['type']
        matl = febtools.material.getclass(matname)
        matlprops = material_map[e.matl_id]['properties']
        F = e.f(r, u)
        p = matl.pstress(F, matlprops) # 1st Piola-Kirchoff stress
        dudx = e.dinterp(r, u)
        dudx1 = dudx[:,0]
        w = matl.w(F, matlprops) # strain energy
        dqdx = e.dinterp(r, q) # 1 x 2 or 1 x 3
        return -w * dqdx[0] + sum(p[i][j] * dudx[i,0] * dqdx[j] 
                                 for i in xrange(len(r))
                                 for j in xrange(len(r)))
    j = 0
    for e in elements:
        j += e.integrate(integrand, u, q, material_map)
    return j

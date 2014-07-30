# -*- coding: utf-8 -*-
"""Functions for conveniently selecting nodes and elements.

"""
import febtools as feb
import numpy as np

tol = np.finfo('float').eps

class ElementSelection:
    def __init__(self, mesh, elements=None):
        """Create a selection of elements.

        mesh := the `mesh` to which the elements belong

        elements := A `set` of elements. Optional. If unset, all
        elements in the mesh are selected.

        This class permits chaining of selection functions, many of
        which need a reference to the parent mesh in addition to the
        element selection.

        """
        self.mesh = mesh
        if elements is None:
            self.elements = set(mesh.elements)
        else:
            self.elements = elements

def corner_nodes(mesh):
    """Return ids of corner nodes.

    """
    ids = [i for i in xrange(len(mesh.nodes))
           if len(mesh.elem_with_node[i]) == 1]
    return ids

def surface_faces(mesh):
    """Return surface faces.

    """
    # Pick a node to start. Nodes with minimum/maximum coordinate
    # values must be surfaces nodes.
    i, j, k = np.argmin(mesh.nodes, axis=0)
    # Advance across the surface with a "front" of active nodes
    surf_faces = set()
    adv_front = set([i])
    processed_nodes = set()
    while adv_front:
        candidates = (f for i in adv_front
                      for f in mesh.faces_with_node[i])
        on_surface = (f for f in candidates
                      if len(adj_faces(mesh, f, mode='face')) == 0)
        surf_faces.update(on_surface)
        processed_nodes.update(adv_front)
        adv_front = set.difference(set([i for f in surf_faces
                                        for i in f.ids]),
                                   processed_nodes)
    return surf_faces

def bisect(elements, v, axis=np.array([0, 0, 1])):
    """Return elements on one side of of a plane.

    v := The distance along `axis` at which to bisect.

    axis := Must coincide with the cartesian coordinate system;
    i.e. two values must be zero and the third nonzero.  Elements
    intersecting the slice plane or on the side of the bisection that
    `axis` points at are returned.

    """
    # figure out which axis we're using
    idx = np.array(axis).nonzero()[0]
    assert len(idx) == 1
    iax = idx[0]
    sgn = np.sign(axis[iax])
    # Perform bisection.  For each element, there are three
    # possibilities: (1) all nodes below bisection → miss, (2) all
    # nodes above bisection → hit, or (3) some nodes above and some
    # below → hit.  Therefore, if any node is above the bisection, the
    # element should be included.
    if sgn > 0:
        eset = [e for e in elements
                if np.any(e.nodes[:,iax] > v)]
    else:
        eset = [e for e in elements
                if np.any(e.nodes[:,iax] < v)]
    return set(eset)

def element_slice(elements, v, extent=tol, axis=np.array([0, 0, 1])):
    """Return a slice of elements.

    v := The distance along `axis` at which the slice plane is
    located.

    axis := A normal vector to the slice plane.  Must coincide with
    the cartesian coordinate system; i.e. two values must be zero and
    the third nonzero.

    Any element within +/- `extent` of `v` along `axis` is included in
    the slice.  The default extent is the floating point precision.
    Therefore, if the selection plane coincides with a node, the
    elements on both sides of the plane will be included.

    """
    # figure out which axis we're using
    idx = np.array(axis).nonzero()[0]
    assert len(idx) == 1
    iax = idx[0]
    axis = np.abs(axis)
    # Select all above lower bound
    v1 = v - extent
    elements = bisect(elements, v1, axis)
    # Select all below upper bound
    v2 = v + extent
    elements = bisect(elements, v2, -axis)
    return set(elements)

def expand_element_set(superset, subset, n):
    """Grow element selection by n elements.

    subset := The growing selection.

    superset := The set of elements that are candidates for growing
    the subset.

    """
    subset = set(subset)
    inactive_nodes = set([])
    active_nodes = set([i for e in subset for i in e.ids])
    candidates = set(superset) - subset
    for iring in xrange(n):
        # Find adjacent elements
        adjacent = set(e for e in candidates
                       if any(i in active_nodes for i in e.ids))
        # Grow the subset
        subset = subset | adjacent
        # Inactivate former boundary nodes
        inactive_nodes.update(active_nodes)
        # Get new boundary (active) nodes
        nodes = set(i for e in adjacent for i in e.ids)
        active_nodes = nodes - inactive_nodes
        # Update list of candidates
        candidates = candidates - adjacent
    return subset

def faces_by_normal(elements, normal, delta=10*tol):
    """Return all faces with target normal.

    """
    target = 1.0 - delta
    faces = []
    for e in elements:
        for n, f in zip(e.face_normals(), e.faces()):
            if np.dot(n, normal) > target:
                faces.append(f)
    return faces

def elements_with_face(mesh, face):
    """Return elements connected to a face.

    The face is represented as a sequence of nodes.  The order of
    nodes does not matter here.

    """
    face = frozenset(face)
    all_faces = set(frozenset(e.faces()) for e in mesh.elements)
    raise NotImplemented

def face_set(mesh, face, angle=30):
    """Select all adjacent faces with similar normals.

    """
    faces = adj_faces(mesh, face, mode='edge')
    raise NotImplemented

def adj_faces(mesh, face, mode='all'):
    """Return faces connected to a face.

    mode : {'all', 'edge', 'face'}
        The type of adjacency desired. Specifying 'edge' returns only
        faces which share an edge with the input face. Specifying
        'face' returns only faces that share every node with the input
        face.

    """
    nc_faces = [mesh.faces_with_node[i] for i in face.ids]
    # ^ faces sharing a node
    fc_faces = set.intersection(*nc_faces) - set([face])
    # ^ other faces sharing all nodes
    if mode == 'face':
        return fc_faces
    edges = [(i, i + 1) for i in xrange(len(face.ids) - 1)]
    edges.append((len(face.ids) - 1, 0))
    ec_faces = set.union(*[set.intersection(nc_faces[i1], nc_faces[i2])
                           for i1, i2 in edges]) - fc_faces - set([face])
    # ^ other faces sharing two nodes
    if mode == 'edge':
        return ec_faces
    elif mode == 'all':
        return set.union(fc_faces, ec_faces)

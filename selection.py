# -*- coding: utf-8 -*-
"""Functions for conveniently selecting nodes and elements.

"""
import operator

import numpy as np

import febtools as feb
from febtools import _canonical_face

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
        candidate_faces = (f for i in adv_front
                           for e in mesh.elem_with_node[i]
                           for f in e.faces()
                           if i in f)
        on_surface = [f for f in candidate_faces
                      if len(adj_faces(f, mesh, mode='face')) == 0]
        surf_faces.update(on_surface)
        processed_nodes.update(adv_front)
        adv_front = set.difference(set([i for f in surf_faces
                                        for i in f]),
                                   processed_nodes)
    return surf_faces

def bisect(elements, p, v):
    """Return elements on one side of of a plane.

    p := A point (x, y, z) on the plane.

    v := A vector (vx, vy, vz) normal to the plane.

    The elements reterned are those either intersected by the cut
    plane or on the side of the plane towards which `v` points.

    """
    # sanitize inputs
    v = np.array(v)
    p = np.array(p)
    # find distance from plane
    def on_pside(e, p=p, v=v):
        """Returns true if element touches (or intersects) plane.

        """
        dpv = np.dot(p, v)
        d = np.dot(e.nodes, v)
        return any(d >= dpv)
    eset = [e for e in elements if on_pside(e)]
    return set(eset)

def element_slice(elements, v, extent=tol, axis=(0, 0, 1)):
    """Return a slice of elements.

    v := The distance along `axis` at which the slice plane is
    located.

    axis := A normal vector to the slice plane.  Must coincide with
    the cartesian coordinate system; i.e. two values must be 0 and
    the third 1.

    Any element within +/- `extent` of `v` along `axis` is included in
    the slice.  The default extent is the floating point precision.
    Therefore, if the selection plane coincides with a node, the
    elements on both sides of the plane will be included.

    """
    # sanitize inputs
    axis = np.abs(np.array(axis), dtype=np.float)
    # figure out which axis we're using
    idx = np.array(axis).nonzero()[0]
    assert len(idx) == 1
    iax = idx[0]
    # Select all above and intersecting lower bound
    v1 = v - extent
    elements = bisect(elements, p=v1 * axis, v=axis)
    # Select all below and intersecting upper bound
    v2 = v + extent
    elements = bisect(elements, p=v2 * axis, v=-axis)
    return set(elements)

def e_grow(selection, candidates, n):
    """Grow element selection by n elements.

    seed := The growing selection.

    candidates := The set of elements that are candidates for growing
    the seed.

    """
    seed = set(selection)
    inactive_nodes = set([])
    active_nodes = set([i for e in seed for i in e.ids])
    candidates = set(candidates) - seed
    for iring in xrange(n):
        # Find adjacent elements
        adjacent = set(e for e in candidates
                       if any(i in active_nodes for i in e.ids))
        # Grow the seed
        seed = seed | adjacent
        # Inactivate former boundary nodes
        inactive_nodes.update(active_nodes)
        # Get new boundary (active) nodes
        nodes = set(i for e in adjacent for i in e.ids)
        active_nodes = nodes - inactive_nodes
        # Update list of candidates
        candidates = candidates - adjacent
    return seed

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

def face_set(mesh, face, angle=30):
    """Select all adjacent faces with similar normals.

    """
    faces = adj_faces(face, mesh, mode='edge')
    raise NotImplemented

def adj_faces(face, mesh, mode='all'):
    """Return faces connected to a face.

    face : tuple of integers
        The face for which adjacent faces will be identified, defined
        by the ids of its nodes.

    mode : {'all', 'edge', 'face'}
        The type of adjacency desired. Specifying 'edge' returns only
        faces which share an edge with the input face. Specifying
        'face' returns only faces that share every node with the input
        face.

    """

    def overlap(a, b):
        """Return the type of adjacency between two faces.

        """
        o = len(set(a) & set(b))
        return o

    face = tuple(face)
    face = _canonical_face(face)

    # faces sharing at least one node, sans the queried face
    nc_faces = set([b for i in face
                    for b in mesh.faces_with_node(i)
                    if b != face])

    if mode == 'all':
        return nc_faces
    elif mode == 'edge':
        return [b for b in nc_faces if overlap(face, b) == 2]
    elif mode == 'face':
        n = len(face)
        return [b for b in nc_faces if overlap(face, b) == n]

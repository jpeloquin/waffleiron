# -*- coding: utf-8 -*-
"""Functions for conveniently selecting nodes and elements.

"""
import febtools as feb
import numpy as np

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

def elements_with_face(mesh, face):
    """Return elements connected to a face.

    The face is represented as a sequence of nodes.  The order of
    nodes does not matter here.

    """
    face = frozenset(face)
    all_faces = set(frozenset(e.faces()) for e in mesh.elements)
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

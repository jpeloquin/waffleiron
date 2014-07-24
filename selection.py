# -*- coding: utf-8 -*-
"""Functions for conveniently selecting nodes and elements.

"""
import febtools as feb

def corner_nodes(mesh):
    """Return ids of corner nodes.

    """
    ids = [i for i in xrange(len(mesh.nodes))
           if len(mesh.elem_with_node[i]) == 1]
    return ids

def surface_faces(mesh):
    """Return surface faces.

    """
    # Find how many elements contain each face
    nconnected = {}
    for e in mesh.elements:
        for f in e.faces():
            k = frozenset(f)
            nconnected[k] = nconnected.setdefault(k, 0) + 1
    surface_faces = [f for f, n in nconnected.iteritems()
                     if n == 1]
    return surface_faces

def elements_with_face(mesh, face):
    """Return elements connected to a face.

    The face is represented as a sequence of nodes.  The order of
    nodes does not matter here.

    """
    face = frozenset(face)
    all_faces = set(frozenset(e.faces()) for e in mesh.elements)
    raise NotImplemented

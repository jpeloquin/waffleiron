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

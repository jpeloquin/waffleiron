import numpy as np
import febtools as feb

def zstack(mesh, zcoords):
    """Stack a 2d mesh in the z direction to make a 3d mesh.

    Arguments
    ---------
    zcoords -- The z-coordinate of each layer of nodes in the stacked
    mesh.  The number of element layers will be one less than the
    length of zcoords.

    Material properties are preserved.  Boundary conditions are not.

    """
    # Create 3d node list
    nodes = []
    for z in zcoords:
        node_layer = [(pt[0], pt[1], z) for pt in mesh.nodes]
        nodes = nodes + node_layer

    # Create elements
    eid = 0
    elements = []
    # Iterate over element layers
    for i in range(len(zcoords) - 1):
        # Iterate over elements in 2d mesh
        for e2d in mesh.elements:
            nids = ([a + i * len(mesh.nodes)
                     for a in e2d.ids] +
                    [a + (i + 1) * len(mesh.nodes)
                     for a in e2d.ids])
            if isinstance(e2d, feb.element.Quad4):
                cls = feb.element.Hex8
            else:
                raise ValueError("Only Quad4 meshes can be used in zstack right now.")
            e3d = cls.from_ids(nids, nodes, mat=e2d.material)
            elements.append(e3d)

    mesh3d = feb.Mesh(nodes=nodes, elements=elements)
    return mesh3d

def merge_node_ids(mesh, nodes):
    """Merge node ids in nodes in `mesh`

    Note that this function does not clean up any orphaned nodes.

    """
    # Adjust references in elements
    nodes = np.array(nodes)
    for e in mesh.elements:
        for j, k in enumerate(e.ids):
            if k in nodes:
                e.ids[j] = nodes[0]

def merge_node_positions(mesh, candidates=None, tol=feb._default_tol):
    """Merge overlapping nodes.

    candidates := list-like of integer node ids.

    """
    if candidates is None:
        candidates = range(len(mesh.nodes))
    for i in candidates:
        # Find nodes in mesh closest to i's position
        x_i = np.array(mesh.nodes[i])
        imatch = set(mesh.find_nearest_nodes(*x_i)) - set([i])
        to_merge = [j for j in imatch if np.linalg.norm(x_i - np.array(mesh.nodes[j])) < tol]
        merge_node_ids(mesh, [i] + to_merge)


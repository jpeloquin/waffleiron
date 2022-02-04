from math import ceil, pi, cos, sin

# Public repo packages
import numpy as np
from numpy.linalg import norm
from shapely.geometry import LineString, Point, Polygon

# Waffleiron modules
import waffleiron as wfl
from .core import FaceSet, NodeSet, _DEFAULT_TOL
from .geometry import pt_series
from .element import Hex8, Quad4
from .model import Mesh


def cylinder(t_radius: tuple, t_height: tuple, nc: int, material=None):
    """Create an FE mesh of a cylinder

    radius := (length, # elements).  The number of elements must be ≥ 1.

    height := (length, # elements).  The number of elements must be ≥ 1.

    nc := int, number of elements along circumference.  Must be ≥ 3.

    Radius is used instead of diameter because the diameter must have
    an even number of elements, whereas the radius can have an odd or
    even number.

    Element spacing is linear.

    The origin is in the center of the cylinder and the height is along the z axis.

    """
    radius, nr = t_radius
    height, nh = t_height
    # Create a mesh of quads, representing one radial slice in the x–z plane.  Points:
    #
    #  A————B  z
    #  |    |  ↑
    #  C————D  · → x
    #
    # with A and C on the central axis of the cylinder.
    A = np.array((0, height / 2))
    B = np.array((radius, height / 2))
    C = np.array((0, -height / 2))
    D = np.array((radius, -height / 2))
    pts_AB = [A + s * (B - A) for s in wfl.math.linspaced(0, 1, n=nr + 1)]
    pts_CD = [C + s * (D - C) for s in wfl.math.linspaced(0, 1, n=nr + 1)]
    pts_AC = [A + s * (C - A) for s in wfl.math.linspaced(0, 1, n=nh + 1)]
    pts_BD = [B + s * (D - B) for s in wfl.math.linspaced(0, 1, n=nh + 1)]
    pane = quadrilateral(pts_AC, pts_BD, pts_CD, pts_AB)
    cylinder = polar_stack_full(pane, nc)
    if material is not None:
        for e in cylinder.elements:
            e.material = material
    # Create named node sets.
    nodes = cylinder.nodes
    z = nodes[:, 2]
    top_nodes = NodeSet(np.where(np.abs(z - height / 2) < _DEFAULT_TOL)[0])
    cylinder.named["node sets"].add("top", top_nodes)
    # Side nodes are coincident with the maximum radius of the cylinder.
    r = np.linalg.norm(nodes[:, 0:2], axis=1)
    side_nodes = NodeSet(np.where(np.abs(r - radius) < _DEFAULT_TOL)[0])
    cylinder.named["node sets"].add("side", side_nodes)
    bottom_nodes = NodeSet(np.where(np.abs(z + height / 2) < _DEFAULT_TOL)[0])
    cylinder.named["node sets"].add("bottom", bottom_nodes)
    exterior_nodes = NodeSet(top_nodes | side_nodes | bottom_nodes)
    cylinder.named["node sets"].add("exterior", exterior_nodes)
    # Create named face sets.  It's annoying to have to define the same surface
    # in terms of both nodes and faces.  Consider improving the situation later.
    top_faces = set()
    for e in cylinder.elements:
        if any([i in top_nodes for i in e.ids]):
            for f in e.faces():
                if all([i in top_nodes for i in f]):
                    top_faces.add(f)
    cylinder.named["face sets"].add("top", FaceSet(top_faces))
    return cylinder


def polar_stack_full(mesh, n):
    """Stack a planar mesh of Quad4 elements about z in a full circle

    mesh := 2D mesh of Quad4 elements.  One edge of the mesh is assumed
    to be on the x^2 + y^2 = 0 centerline.

    n := number of element layers to create.

    The resulting mesh has a core of Penta6 elements surrounded by Hex8 elements.

    TODO: Support input meshes that do not have nodes on the centerline,
    e.g., for creating hollow cylinders.

    """
    # Convert 2D nodes array to 3D in x–z plane
    nodes = np.array([(n[0], 0, n[1]) for n in mesh.nodes])
    # Divide the nodes and elements into centerline and rotating groups
    nids = np.arange(len(nodes))
    on_centerline = np.abs(np.array(nodes)[:, 0]) < _DEFAULT_TOL
    center_nodes = nodes[on_centerline]
    center_nids = nids[on_centerline]
    rotate_nodes = nodes[np.logical_not(on_centerline)]
    rotate_nids = nids[np.logical_not(on_centerline)]
    center_elements = []  # use lists to preserve order
    center_eids = set()
    rotate_elements = []
    center_nids_set = set(center_nids)
    for i, element in enumerate(mesh.elements):
        if set(element.ids) & center_nids_set:
            center_elements.append(element)
            center_eids.add(i)
        else:
            rotate_elements.append(element)
    # Make sure nodes are ordered so their face normal will face out of
    # the 3D element.
    for element in mesh.elements:
        p = nodes[element.faces()[0], :]
        v1 = p[1] - p[0]
        v2 = p[-1] - p[0]
        v3 = wfl.geometry.cross(v1, v2)
        if v3[1] < 0:
            element.ids = element.ids[::-1]
        elif v3[2] == 0:
            raise ValueError(
                f"Input {type(element)} element with node IDs {element.ids} was not in x–y plane."
            )
    # Construct the 3D mesh
    all_nodes = [nodes]
    new_elements = []
    # New nodes will have an index equal to the offset for the
    # corresponding node in the original node layer plus the stride ×
    # the number of element layers.
    offset = len(nodes) - np.cumsum(on_centerline)
    offset[on_centerline] = 0
    # ^ centerline nodes are reused in all layers
    stride = np.zeros(len(nodes), dtype=int)
    stride[~on_centerline] = len(rotate_nids)
    for ilayer in range(0, n):  # i indexes over polar element layers
        # Construct new node layer
        if ilayer != n - 1:
            θ = 2 * pi * (ilayer + 1) / n
            Q = np.array([[cos(θ), -sin(θ), 0], [sin(θ), cos(θ), 0], [0, 0, 1]])
            new_nodes = (Q @ rotate_nodes.T).T
            all_nodes += [new_nodes]
        # Compute node IDs for the new layer of elements
        for eid, element in enumerate(mesh.elements):
            if eid in center_eids:
                # Element is on centerline; list node IDs for Penta6.
                # None of the Penta6 faces follow the Quad4 node
                # ordering, so we need to rearrange nodes.  In the
                # Penta6 elment: Put the node 1 to node 4 line on the
                # centerline, pointing +z.  Nodes 2 and 5 will come from
                # the old layer.
                m = on_centerline[element.ids]
                c_nids = element.ids[m]
                r_nids = element.ids[~m]
                c_nids = c_nids[np.argsort(nodes[c_nids][:, 2])]
                r_nids = r_nids[np.argsort(nodes[r_nids][:, 2])]
                if ilayer == 0:
                    r_nids_last = r_nids
                else:
                    r_nids_last = (
                        r_nids + offset[r_nids] + (ilayer - 1) * stride[r_nids]
                    )
                if ilayer == n - 1:
                    r_nids_next = r_nids
                else:
                    r_nids_next = r_nids + offset[r_nids] + ilayer * stride[r_nids]
                new_elements.append(
                    (
                        wfl.element.Penta6,
                        (
                            c_nids[0],
                            r_nids_last[0],
                            r_nids_next[0],
                            c_nids[1],
                            r_nids_last[1],
                            r_nids_next[1],
                        ),
                    )
                )
            else:
                # Element is off centerline; list node IDs for Hex8
                if ilayer == 0:
                    r_nids_last = element.ids
                else:
                    r_nids_last = (
                        element.ids
                        + offset[element.ids]
                        + (ilayer - 1) * stride[element.ids]
                    )
                if ilayer == n - 1:
                    r_nids_next = element.ids
                else:
                    r_nids_next = (
                        element.ids + offset[element.ids] + ilayer * stride[element.ids]
                    )
                new_elements.append(
                    (wfl.element.Hex8, np.hstack((r_nids_last, r_nids_next)))
                )
    # Construct objects for the new mesh
    nodes = np.vstack(all_nodes)
    elements = []
    for cls, ids in new_elements:
        elements.append(cls.from_ids(ids, nodes))
    mesh = Mesh(nodes, elements)
    return mesh


def zstack(mesh, zcoords):
    """Stack a 2d mesh in the z direction to make a 3d mesh.

    Arguments
    ---------
    zcoords -- The z-coordinate of each layer of nodes in the stacked
    mesh.  The number of element layers will be one less than the
    length of zcoords.

    Material properties are preserved.  Boundary conditions are not.

    """
    # `zstack` is here instead of in util.py because a it's used
    # extensively in this module to build 3D meshes.
    #
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
            nids = [a + i * len(mesh.nodes) for a in e2d.ids] + [
                a + (i + 1) * len(mesh.nodes) for a in e2d.ids
            ]
            if isinstance(e2d, Quad4):
                cls = Hex8
            else:
                raise ValueError("Only Quad4 meshes can be used in zstack right now.")
            e3d = cls.from_ids(nids, nodes, mat=e2d.material)
            elements.append(e3d)

    mesh3d = Mesh(nodes=nodes, elements=elements)
    return mesh3d


def rectangular_prism(length, width, thickness, material=None):
    """Create an FE mesh of a rectangular prism.

    Each key dimension is a tuple of (length, number of elements).
    Element spacing is linear.

    The origin is in the center of the rectangle.

    """
    l = length[0]
    nl = length[1]
    w = width[0]
    nw = width[1]
    t = thickness[0]
    nt = thickness[1]
    # Create rectangle in xy plane
    A = np.array([-l / 2, -w / 2])
    B = np.array([l / 2, -w / 2])
    C = np.array([l / 2, w / 2])
    D = np.array([-l / 2, w / 2])
    AB = pt_series([A, B], nl + 1)
    DC = pt_series([D, C], nl + 1)
    AD = pt_series([A, D], nw + 1)
    BC = pt_series([B, C], nw + 1)
    mesh = quadrilateral(AD, BC, AB, DC)
    # Create rectangular prism
    zi = np.linspace(-t / 2, t / 2, nt + 1)
    mesh = zstack(mesh, zi)
    # Label the faces
    nodes = np.array(mesh.nodes)  # remove when array is default
    # ±x faces
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 0], -l / 2, rtol=0, atol=np.spacing(l / 2)))[0]
    )
    assert len(ns) == (nw + 1) * (nt + 1)
    mesh.named["node sets"].add("−x1 face", ns)
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 0], l / 2, rtol=0, atol=np.spacing(l / 2)))[0]
    )
    assert len(ns) == (nw + 1) * (nt + 1)
    mesh.named["node sets"].add("+x1 face", ns)
    # ±y faces
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 1], -w / 2, rtol=0, atol=np.spacing(w / 2)))[0]
    )
    assert len(ns) == (nl + 1) * (nt + 1)
    mesh.named["node sets"].add("−x2 face", ns)
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 1], w / 2, rtol=0, atol=np.spacing(w / 2)))[0]
    )
    assert len(ns) == (nl + 1) * (nt + 1)
    mesh.named["node sets"].add("+x2 face", ns)
    # ±z faces
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 2], -t / 2, rtol=0, atol=np.spacing(t / 2)))[0]
    )
    assert len(ns) == (nl + 1) * (nw + 1)
    mesh.named["node sets"].add("−x3 face", ns)
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 2], t / 2, rtol=0, atol=np.spacing(t / 2)))[0]
    )
    assert len(ns) == (nl + 1) * (nw + 1)
    mesh.named["node sets"].add("+x3 face", ns)
    # Assign material
    if material is not None:
        for e in mesh.elements:
            e.material = material
    return mesh


def quadrilateral(col1, col2, row1, row2):
    """Mesh a quadrilateral with quad elements.

    Each input variable is a list of (x, y) points specifying node
    locations on the boundary of the mesh domain.  `col1` and `col2` are
    opposite edges, as are `row1` and `row2`.  Row1 must be below (−y)
    row2 and col1 should be left (−x) of col2 in a right hand coordinate
    system.

    """

    def pdist(p1, p2):
        """Distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Ensure col1 lines up with the start of row1
    d_row1p0_col1 = min(pdist(row1[0], col1[i]) for i in [0, -1])
    d_row1p0_col2 = min(pdist(row1[0], col2[i]) for i in [0, -1])
    if d_row1p0_col1 > d_row1p0_col2:
        # col2 intersects the start of row1, so swap col2 and col1
        col1, col2 = col2, col1

    # Ensure both columns start on row1
    if pdist(row1[0], col1[0]) > pdist(row1[0], col1[-1]):
        col1 = col1[::-1]
    if pdist(row1[-1], col2[0]) > pdist(row1[-1], col2[-1]):
        col2 = col2[::-1]

    # Ensure the row2 starts on col1 (parallels row1)
    if pdist(col1[-1], row2[0]) > pdist(col1[-1], row2[-1]):
        row2 = row2[::-1]

    # Compute corresponding normalized path length for each point
    def spts_norm(pts):
        s = [0]
        for i in range(1, len(pts)):
            p0 = np.array(pts[i - 1])
            p1 = np.array(pts[i])
            s.append(s[i - 1] + np.linalg.norm(p1 - p0))
        stot = s[-1]
        snorm = [a / stot for a in s]
        return snorm

    col1_s = spts_norm(col1)
    col2_s = spts_norm(col2)
    row1_s = spts_norm(row1)
    row2_s = spts_norm(row2)

    # Ensure the the node counts are compatible
    assert len(row1) == len(row2)
    assert len(col1) == len(col2)
    nr = len(col1)
    nc = len(row1)

    # For debugging
    #
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.plot(np.array(row1)[:,0], np.array(row1)[:,1], 'k-o')
    # plt.plot(np.array(row2)[:,0], np.array(row2)[:,1], 'k-*')
    # plt.plot(np.array(col1)[:,0], np.array(col1)[:,1], 'r-o')
    # plt.plot(np.array(col2)[:,0], np.array(col2)[:,1], 'r-*')
    # plt.show()

    # Create nodes

    # initialize loop
    prevrow_pts = row1
    elements = []
    nodes = prevrow_pts
    # Loop over rows of nodes, creating one row at a time, from -x2 to
    # +x2.  We already initialized the -x2 most row above.
    for i in range(1, nr):

        # Add interpolated points for this row
        thisrow_pts = [col1[i]]
        v_c = np.array(col2[i]) - np.array(col1[i])
        u_c = v_c / np.linalg.norm(v_c)
        for j in range(1, nc - 1):
            v_r = np.array(row2[j] - row1[j])
            u_v = v_r / np.linalg.norm(v_r)
            ln_r = LineString([col1[i] - v_c, col2[i] + v_c])
            ln_c = LineString([row1[j] - v_r, row2[j] + v_r])
            # ^ extend the lines a little bit to avoid missing an
            # intersection due to finite precision
            pt = ln_r.intersection(ln_c)
            if pt.is_empty:
                raise Exception(
                    "When placing a node, no "
                    "intersection was found between "
                    "lines drawn between gridlines. "
                    "Check the input geometry."
                )
            pt = pt.coords[0]
            thisrow_pts.append(pt)
        thisrow_pts.append(col2[i])
        nodes = nodes + thisrow_pts

    # Stitch elements
    for i in range(1, nr):
        # some setup
        n_base = (i - 1) * nc  # number of nodes already fully stitched
        prevrow_pts = thisrow_pts
        # do the actual stitching
        for j in range(nc - 1):
            e = np.array([0, 1, nc + 1, nc]) + n_base + j
            elements.append(e)

    mesh = Mesh.from_ids(nodes, elements, Quad4)

    return mesh

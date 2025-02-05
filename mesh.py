from math import ceil, pi, cos, sin
import sys

# Public repo packages
import numpy as np
from numpy.linalg import norm
from shapely.geometry import LineString, Point, Polygon

# Waffleiron modules
import waffleiron as wfl
from .core import FaceSet, NodeSet, _DEFAULT_TOL
from .geometry import pt_series
from .element import Hex27, Hex8, Quad4
from .math import linspaced
from .model import Mesh


def cylinder(t_radius: tuple, t_height: tuple, nc: int, bias_h=1, material=None):
    """Create an FE mesh of a cylinder

    :param t_radius: (length, # elements).  The number of elements must be ≥ 1.

    :param t_height: (length, # elements).  The number of elements must be ≥ 1.

    :param nc: number of elements along circumference.  Must be ≥ 3.

    :param bias_h: Bias factor for element spacing along the z-axis (height) of the
    cyclinder.  Elements are ordered such that the top layer is first and the bottom
    layer is last.

    :param material: Material to assign to the cylinder's elements.

    Element spacing is linear.

    The origin is in the center of the cylinder and the height is along the z axis.

    """
    # Radius is used (instead of diameter) to simplify validation. A diameter with an
    # odd number of elements would be invalid, but a radius can have an even or odd
    # number of elements.
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
    pts_AB = [A + s * (B - A) for s in wfl.math.linspaced(0, 1, nr + 1)]
    pts_CD = [C + s * (D - C) for s in wfl.math.linspaced(0, 1, nr + 1)]
    pts_AC = [A + s * (C - A) for s in wfl.math.x_biasfactor(0, 1, nh, bias_h)]
    pts_BD = [B + s * (D - B) for s in wfl.math.x_biasfactor(0, 1, nh, bias_h)]
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


def label_rectangular_prism(mesh: Mesh, bounds=None):
    """Add standard node set labels to a rectangular prism mesh"""
    nodes = np.array(mesh.nodes)  # remove when array is default
    if bounds is None:
        xmin = np.min(nodes, axis=0)
        xmax = np.max(nodes, axis=0)
        bounds = [(xmin[0], xmax[0]), (xmin[1], xmax[1]), (xmax[2], xmax[2])]
    # Label the faces
    # ±x faces
    ns = NodeSet(
        np.nonzero(
            np.isclose(
                nodes[:, 0], bounds[0][0], rtol=0, atol=abs(np.spacing(bounds[0][0]))
            )
        )[0]
    )
    mesh.named["node sets"].add("-x1 face", ns)
    ns = NodeSet(
        np.nonzero(
            np.isclose(
                nodes[:, 0], bounds[0][1], rtol=0, atol=abs(np.spacing(bounds[0][1]))
            )
        )[0]
    )
    mesh.named["node sets"].add("+x1 face", ns)
    # ±y faces
    ns = NodeSet(
        np.nonzero(
            np.isclose(
                nodes[:, 1], bounds[1][0], rtol=0, atol=abs(np.spacing(bounds[1][0]))
            )
        )[0]
    )
    mesh.named["node sets"].add("-x2 face", ns)
    ns = NodeSet(
        np.nonzero(
            np.isclose(
                nodes[:, 1], bounds[1][1], rtol=0, atol=abs(np.spacing(bounds[1][1]))
            )
        )[0]
    )
    mesh.named["node sets"].add("+x2 face", ns)
    # ±z faces
    ns = NodeSet(
        np.nonzero(
            np.isclose(
                nodes[:, 2], bounds[2][0], rtol=0, atol=abs(np.spacing(bounds[2][0]))
            )
        )[0]
    )
    mesh.named["node sets"].add("-x3 face", ns)
    ns = NodeSet(
        np.nonzero(
            np.isclose(
                nodes[:, 2], bounds[2][1], rtol=0, atol=abs(np.spacing(bounds[2][1]))
            )
        )[0]
    )
    mesh.named["node sets"].add("+x3 face", ns)


def polar_stack_full(mesh, n):
    """Stack a planar mesh of Quad4 elements in x–z in a full circle around z

    mesh := 2D mesh of Quad4 elements.  One edge of the mesh is assumed to be on the x^2
    + y^2 = 0 centerline.

    n := number of element layers to create.

    The resulting mesh has a core of Penta6 elements surrounded by Hex8 elements.

    TODO: Support input meshes that do not have nodes on the centerline, e.g., for
    creating hollow cylinders.

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


def points_on_line(A, B, s):
    """Return points at normalized arc length positions s along line AB

    The points are returned as numpy arrays.

    """
    A = np.array(A)
    B = np.array(B)
    if min(s) < 0:
        raise ValueError(f"s must be in [0, 1].  Min value was {min(s)}.")
    if max(s) > 1:
        raise ValueError(f"s must be in [0, 1].  Max value was {max(s)}.")
    return [A + si * (B - A) for si in s]


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


def rectangular_prism(
    n,
    element_type,
    bounds=((-1, 1), (-1, 1), (-1, 1)),
    bias_fun=(linspaced, linspaced, linspaced),
    material=None,
):
    """Mesh an axis-aligned rectangular prism domain

    :param n: [nx, ny, nz], where nx is the element count along length (x), ny is the
    element count along the width (y), and nz is the same along the height (z).

    :param element_type: "Hex8" and "Hex27" are currently supported.

    :param bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) extent of mesh.

    :bias_fun: Three functions of the form `f(A, B, n)` where `A` and `B` are the
    endpoints of the line segment AB and the return value is a sequence of points along
    AB with spacing equal to the desired element spacing.  `f` will be called for each
    of the three axes with `A` corresponding to the lower bound listed in `bounds` and
    `B` corresponding to the upper bound.

    Bounds are accepted as arguments to avoid having to apply arithmetic to shift or
    scale the mesh after creation.  This avoids loss of precision in the node positions.

    """
    # Element type
    if isinstance(element_type, str):
        element_type = getattr(sys.modules[__name__], element_type)
    if element_type == Hex8:
        mesh = rectangular_prism_hex8(n, bounds, bias_fun)
    elif element_type == Hex27:
        mesh = rectangular_prism_hex27(n, bounds, bias_fun)
    else:
        raise ValueError(f"Element type '{element_type}' not supported.")
    # Bounds
    for i, b in enumerate(bounds):
        if b[1] - b[0] <= 0:
            raise ValueError(f"{bounds=} has length <= 0 along axis {i}.")
    # Assign material
    if material is not None:
        for e in mesh.elements:
            e.material = material
    return mesh


def rectangular_prism_hex8(
    n, bounds=((-1, 1), (-1, 1), (-1, 1)), bias_fun=(linspaced, linspaced, linspaced)
):
    """Return a Hex8 mesh of a rectangular prism"""
    ne = np.array(n)
    nn = np.array(n) + 1
    nodes = np.array(
        np.meshgrid(
            bias_fun[0](bounds[0][0], bounds[0][1] - bounds[0][0], nn[0]),
            bias_fun[1](bounds[1][0], bounds[1][1] - bounds[1][0], nn[1]),
            bias_fun[2](bounds[2][0], bounds[2][1] - bounds[2][0], nn[2]),
            indexing="ij",
        )
    )  # first index over xyz
    ids_for_element = np.full(ne, None)
    for i in range(ne[0]):
        for j in range(ne[1]):
            for k in range(ne[2]):
                id3 = np.array(
                    [
                        [i, j, k],  # 1
                        [i + 1, j, k],  # 2
                        [i + 1, j + 1, k],  # 3
                        [i, j + 1, k],  # 4
                        [i, j, k + 1],  # 5
                        [i + 1, j, k + 1],  # 6
                        [i + 1, j + 1, k + 1],  # 7
                        [i, j + 1, k + 1],  # 8
                    ]
                )
                id1 = [np.ravel_multi_index(t, nodes.shape[1:], order="C") for t in id3]
                ids_for_element[i, j, k] = id1
    mesh = Mesh.from_ids(
        nodes.reshape(
            (3, -1),
        ).T,
        ids_for_element.reshape(-1),
        Hex8,
    )
    # Label the faces
    label_rectangular_prism(mesh, bounds)
    assert len(mesh.named["node sets"].obj("−x1 face")) == (nn[1] * nn[2])
    assert len(mesh.named["node sets"].obj("+x1 face")) == (nn[1] * nn[2])
    assert len(mesh.named["node sets"].obj("−x2 face")) == (nn[0] * nn[2])
    assert len(mesh.named["node sets"].obj("+x2 face")) == (nn[0] * nn[2])
    assert len(mesh.named["node sets"].obj("−x3 face")) == (nn[0] * nn[1])
    assert len(mesh.named["node sets"].obj("+x3 face")) == (nn[0] * nn[1])
    return mesh


def rectangular_prism_hex27(
    n, bounds=[(-1, 1), (-1, 1), (-1, 1)], bias_fun=[linspaced, linspaced, linspaced]
):
    """Return a Hex27 mesh of a rectangular prism"""
    ne = np.array(n)
    nn = 2 * ne + 1  # total number of nodes in each direction
    nodes = np.array(
        np.meshgrid(
            bias_fun[0](bounds[0][0], bounds[0][1] - bounds[0][0], nn[0]),
            bias_fun[1](bounds[1][0], bounds[1][1] - bounds[1][0], nn[1]),
            bias_fun[2](bounds[2][0], bounds[2][1] - bounds[2][0], nn[2]),
            indexing="ij",
        )
    )  # first index over xyz
    ids_for_element = np.full(ne, None)
    for i in range(ne[0]):
        for j in range(ne[1]):
            for k in range(ne[2]):
                id3 = np.array(
                    [
                        [i * 2, j * 2, k * 2],  # 1
                        [i * 2 + 2, j * 2, k * 2],  # 2
                        [i * 2 + 2, j * 2 + 2, k * 2],  # 3
                        [i * 2, j * 2 + 2, k * 2],  # 4
                        [i * 2, j * 2, k * 2 + 2],  # 5
                        [i * 2 + 2, j * 2, k * 2 + 2],  # 6
                        [i * 2 + 2, j * 2 + 2, k * 2 + 2],  # 7
                        [i * 2, j * 2 + 2, k * 2 + 2],  # 8
                        [i * 2 + 1, j * 2, k * 2],  # 9
                        [i * 2 + 2, j * 2 + 1, k * 2],  # 10
                        [i * 2 + 1, j * 2 + 2, k * 2],  # 11
                        [i * 2, j * 2 + 1, k * 2],  # 12
                        [i * 2 + 1, j * 2, k * 2 + 2],  # 13
                        [i * 2 + 2, j * 2 + 1, k * 2 + 2],  # 14
                        [i * 2 + 1, j * 2 + 2, k * 2 + 2],  # 15
                        [i * 2, j * 2 + 1, k * 2 + 2],  # 16
                        [i * 2, j * 2, k * 2 + 1],  # 17
                        [i * 2 + 2, j * 2, k * 2 + 1],  # 18
                        [i * 2 + 2, j * 2 + 2, k * 2 + 1],  # 19
                        [i * 2, j * 2 + 2, k * 2 + 1],  # 20
                        [i * 2 + 1, j * 2, k * 2 + 1],  # 21
                        [i * 2 + 2, j * 2 + 1, k * 2 + 1],  # 22
                        [i * 2 + 1, j * 2 + 2, k * 2 + 1],  # 23
                        [i * 2, j * 2 + 1, k * 2 + 1],  # 24
                        [i * 2 + 1, j * 2 + 1, k * 2],  # 25
                        [i * 2 + 1, j * 2 + 1, k * 2 + 2],  # 26
                        [i * 2 + 1, j * 2 + 1, k * 2 + 1],  # 27
                    ]
                )
                id1 = [np.ravel_multi_index(t, nodes.shape[1:], order="C") for t in id3]
                ids_for_element[i, j, k] = id1
    mesh = Mesh.from_ids(
        nodes.reshape(
            (3, -1),
        ).T,
        ids_for_element.reshape(-1),
        Hex27,
    )
    # Label the faces
    label_rectangular_prism(mesh, bounds)
    assert len(mesh.named["node sets"].obj("-x1 face")) == (nn[1]) * (nn[2])
    assert len(mesh.named["node sets"].obj("+x1 face")) == (nn[1]) * (nn[2])
    assert len(mesh.named["node sets"].obj("-x2 face")) == (nn[0]) * (nn[2])
    assert len(mesh.named["node sets"].obj("+x2 face")) == (nn[0]) * (nn[2])
    assert len(mesh.named["node sets"].obj("-x3 face")) == (nn[0]) * (nn[1])
    assert len(mesh.named["node sets"].obj("+x3 face")) == (nn[0]) * (nn[1])
    return mesh


def quadrilateral(col1, col2, row1, row2):
    """Mesh a quadrilateral with quad elements.

    Each input variable is a list of (x, y) points specifying node
    locations on the boundary of the mesh domain.  `col1` and `col2` are
    opposite edges, as are `row1` and `row2`.  Row1 must be below (-y)
    row2 and col1 should be left (-x) of col2 in a right hand coordinate
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

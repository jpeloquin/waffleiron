import numpy as np
from scipy.optimize import fmin

import waffleiron as wfl
from waffleiron.exceptions import InvalidConditionError
from waffleiron.geometry import cross


def elem_obj(element, nodes, eid=None):
    """Returns an Element object from node and element tuples.

    element := the indices of this element's nodes
    nodes := the global list of node coordinates

    """
    n = len(element)
    if n == 3:
        etype = Tri3
    elif n == 4 and (len(nodes[0]) == 2 or all([x[2] == 0.0 for x in nodes])):
        etype = Quad4
    elif n == 8:
        etype = Hex8
    else:
        s = "{} node element not recognized".format(n)
        raise Exception(s)
    return etype(element, nodes)


class Element:
    """Data and metadata for an element.

    Attributes
    ----------
    nodes := Nodal positions in reference configuration.  The order follows FEBio
    convention.

    ids := (optional) Nodal indices into `mesh.nodes`

    material := (optional) Material object instance.

    properties := A dictionary of nodal values.  Expected keys are
    'displacement' (indexing a list of displacement vectors).

    Notes
    -----
    When using 2d elements, they should coincide with the xy plane.
    The current implementation of the methods cannot do anything
    useful with the z dimension.  This will be fixed.

    """

    def __init__(self, nodes, material=None):
        """Create an element from a list of nodes.

        Parameters
        ----------
        nodes : n × 3 array-like or n × 1 array-like
            Coordinates for n nodes.  The second dimension is x, y, and z.

        """
        self._ids = np.arange(len(nodes))
        # ^ Indices of nodes.  For a standalone element, the standard indices.  For
        # an element belonging to a mesh, these will be updated to index into the
        # mesh's node list.
        self._mesh = None  # None unless added to a Mesh
        self.basis = None
        self.material = material
        self.properties = {"displacement": np.array([(0, 0, 0) for i in nodes])}
        # Nodal coordinates
        self._nodes = np.array(nodes)
        assert self.nodes.shape[1] >= 2

    @classmethod
    def from_ids(cls, ids, nodelist, mat=None):
        """Create an element from nodal indices."""
        nodes = np.array([nodelist[i] for i in ids])
        element = cls(nodes, mat)
        element.ids = ids
        return element

    def apply_property(self, label, values):
        """Apply nodal properties."""
        assert len(values) == len(self.nodes)
        values = np.array(values)
        self.properties[label] = values

    @property
    def ids(self):
        return self._ids

    @ids.setter
    def ids(self, value):
        self._ids = np.array(value)

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value
        self._nodes = None  # will access parent mesh from now on

    @property
    def nodes(self):
        if self.mesh is not None:
            return self.mesh.nodes[self.ids]
        else:
            return self._nodes

    @nodes.setter
    def nodes(self, value):
        if self.mesh is None:
            self._nodes = np.array(value)
        else:
            self.mesh.nodes[self._ids] = np.array(value)

    def x(self, config="reference"):
        """Nodal positions in reference or deformed configuration.

        Index 1 := node number.

        Index 2 := coordinate.

        """
        if config == "reference":
            x = self.nodes
        elif config == "deformed":
            x = self.nodes + self.properties["displacement"]
        else:
            raise Exception(
                'Value "{}" for config not recognized.  Use "reference" or "deformed"'.format(
                    config
                )
            )
        return x

    def interp(self, r, prop="displacement"):
        """Interpolate node-valued data at r.

        values := A list with a 1:1 mapping to the list of nodes in
        the mesh.  The list elements can be scalar or vector valued
        (but must be consistent).

        """
        if prop == "position":
            v = self.x()
        else:
            v = self.properties[prop]  # nodal values
        return np.dot(v.T, self.N(*r))

    def dinterp(self, r, prop="displacement"):
        """Return d/dx of node-valued data at natural basis point r

        The node-valued data may be scalar, vector, or tensor.

        The returned array indexes into the spatial derivative
        denomator with its last dimension.  The other indexes are the
        same as in the input array.

        Given a scalar, the returned array is 1 × 3.

        Given a n-length vector, the returned array is n × 3.

        Given an n × m matrix, the returned array is n × m × 3.

        And so on for higher order tensors.

        """
        # nodal values
        if type(prop) is str:
            nodal_v = self.properties[prop]
        else:
            nodal_v = prop

        j = self.j(r)
        jinv = np.linalg.pinv(j)
        ddr = np.vstack(self.dN(*r))

        ishape = nodal_v.shape[1:]
        if not ishape:
            ishape = (1,)
        oshape = ishape + (j.shape[0],)  # allow true 2d elements

        flat_v = np.array([a.ravel() for a in nodal_v])
        dvdr = np.dot(ddr.T, flat_v).T
        dvdx = np.dot(dvdr, jinv)

        # Undo raveling
        out = dvdx.reshape(oshape)

        return out.squeeze()

    def ddinterp(self, r, prop="displacement"):
        if type(prop) is str:
            nodal_v = self.properties[prop]  # nodal values
        else:
            nodal_v = prop

        j = self.j(r)
        jinv = np.linalg.pinv(j)
        derivatives = self.ddN(*r)

        # shape of nodal arrays
        ishape = nodal_v.shape[1:]
        if not ishape:
            ishape = (1,)
        # desired shape of output array
        oshape = ishape + (j.shape[0], j.shape[0])
        #                  ^ allow true 2d elements

        # Flatten input
        flat_v = np.array([a.ravel() for a in nodal_v])
        dvdr = np.dot(derivatives.T, flat_v).T
        dvdx = np.dot(jinv.T, np.dot(dvdr, jinv))

        # Right now, dv[i,j,...] / dx[u, v] has its axis in order of
        # u, flattened i,j,... , and v.  The i and j indices must
        # be unflattened and k moved to the second to last position.

        # Unflatten output
        out = dvdx.reshape(oshape)
        out = np.rollaxis(out, 0, start=-1)

        return out.squeeze()

    def f_avg(self):
        """Return F tensor averaged across integration points."""
        return np.mean([self.f(self.gloc[i]) for i in range(len(self.gloc))], axis=0)

    def f(self, r):
        """Calculate F tensor (convenience function).

        r := coordinate vector in element's natural basis

        """
        dudx = self.dinterp(r, prop="displacement")
        F = dudx + np.eye(3)
        return F

    def j(self, r, config="reference"):
        """Jacobian matrix (∂x_i/∂r_j) evaluated at r"""
        ddr = self.dN(*r)
        ddr = np.vstack(self.dN(*r))
        J = np.dot(np.array(self.nodes).T, ddr)
        return J

    def integrate(self, fn, *args, **kwargs):
        """Integrate a function over the element.

        f := The function to integrate.  Must be callable as `f(e,
            r)`, with `e` being this element object instance and `r`
            being 3-element array-like specifing the natural basis
            coordinate at which f is evaluated.

        """
        return sum(
            (
                fn(self, r, *args, **kwargs) * self.jdet(r) * w
                for r, w in zip(self.gloc, self.gwt)
            )
        )

    def centroid(self, config="reference"):
        """Centroid of element."""
        x = self.x(config)
        return self.interp((0, 0, 0), "position")

    def faces(self):
        """Return the faces of this element.

        A face is represented by a tuple of node ids oriented such
        that the cross product returns an outward-pointing normal.

        """
        if self.ids is not None:
            faces = tuple(tuple(self.ids[i] for i in f) for f in self.face_nodes)
        else:
            faces = self.face_nodes
        faces = [wfl._canonical_face(f) for f in faces]
        return faces

    def face_normals(self, config="reference"):
        """List of face normals"""
        normals = []
        # Iterate over faces
        for f in self.faces():
            normals.append(wfl.geometry.face_normal(f, self.mesh))
        return normals

    def faces_with_node(self, node_id):
        """Indices of faces that include a node.

        The node id here is local to the element.

        """
        return [i for i, f in enumerate(self.face_nodes) if node_id in f]

    def to_natural(self, pt, config="reference"):
        """Return natural coordinates for pt = (x, y, z)"""
        pt = np.array(pt)
        x = self.x(config)
        x0 = np.dot(self.N(*[0] * self.r_n), x)
        v = pt - x0
        j = self.j([0] * self.r_n)
        jinv = np.linalg.pinv(j)
        r = np.dot(jinv, v)

        def fn(r, e=self, p=pt):
            return np.sum((np.array(p) - e.interp(r, prop="position")) ** 2.0)

        # def dfn(r, e=self, p=pt):
        #     sign = np.sign(np.array(pt)
        #                     - e.interp(r, prop='position'))
        #     sign = np.atleast_2d(sign).T
        #     # make sign column vector so it is broadcast over the
        #     # columns of dvdr
        #     ddr = np.vstack(self.dN(*r))
        #     dvdr = np.dot(ddr.T, self.x()).T
        #     return np.sum(sign * -dvdr, axis=0)

        r = fmin(fn, r, disp=False)
        return r

    def tstress(self, r):
        F = self.f(r)
        if self.basis is None:
            return self.material.tstress(F)
        else:
            # TODO: Assumes 2-tensor material orientation, but in a
            # fully displacement-constrained simulation, users can use
            # bare fibers as the material, which take 1-tensor (vector)
            # material orientation.
            Q = self.basis
            σ_world = self.material.tstress(F @ Q)
        return σ_world


class Element3D(Element):
    """Class for 3D elements."""

    is_planar = False

    def jdet(self, r, config="reference"):
        """Calculate determinant of the R3 → R3 jacobian."""
        return np.linalg.det(self.j(r, config))


class Element2D(Element):
    """Class for shell (2D) elements.

    All shell elements should inherit from this class.

    """

    is_planar = True

    def jdet(self, r, config="reference"):
        """Calculate determinant of the R3 → R2 jacobian."""
        j = self.j(r, config)
        n = np.cross(j[:, 0], j[:, 1])
        try:
            return n[2]
        except IndexError:
            # i.e if 2d vectors were used
            return n

    def edges(self):
        """Return the edges of this element.

        A edges is represented by a tuple of node ids oriented such
        that the cross product returns an outward-pointing normal.

        """
        if self.ids is not None:
            edges = tuple(tuple(self.ids[i] for i in edge) for edge in self.edge_nodes)
        else:
            edges = self.edge_nodes
        return edges

    def edges_with_node(self, node_id):
        """Indices of edges that include node id."""
        return [i for i, l in enumerate(self.edge_nodes) if node_id in l]

    def edge_normals(self, config="reference"):
        """Return list of edge normals.

        The edge normals are constrained to lie in the same plane as
        the element.  The normals point outward.

        """
        points = self.x(config)
        normals = []
        # Iterate over edges
        for l in self.edge_nodes:
            v = points[l[1]] - points[l[0]]
            face_normal = self.face_normals()[0]
            normals.append(cross(v, face_normal))
        return normals


class Tri3(Element2D):
    """Functions for tri3 elements."""

    n = 3
    r_n = 2  # number of natural basis parameters
    node_connectivity = [[1, 2], [0, 2], [1, 0]]

    feb_name = "tri3"

    # oriented so positive normal follows node ordering convention
    face_nodes = [[0, 1, 2]]

    def __init__(self, *args, **kwargs):
        super(Tri3, self).__init__(*args, **kwargs)
        self.properties["thickness"] = (1.0, 1.0, 1.0)

    def centroid(self, config="reference"):
        """Centroid of element."""
        x = self.x(config)
        r = (1.0 / 3.0, 1.0 / 3.0)
        c = np.dot(x.T, self.N(*r))
        return c

    @staticmethod
    def N(r, s, t=None):
        """Shape functions."""
        n = [0.0] * 3
        n[0] = 1.0 - r - s
        n[1] = r
        n[2] = s
        return n

    @staticmethod
    def dN(r, s, t=None):
        """Shape function 1st derivatives."""
        dn = [np.zeros(2) for i in range(3)]
        # d/dr
        dn[0][0] = -1.0
        dn[1][0] = 1.0
        dn[2][0] = 0.0
        # d/ds
        dn[0][1] = -1.0
        dn[1][1] = 0.0
        dn[2][1] = 1.0
        return dn


class Hex8(Element3D):
    """8-node (trilinear) hexahedral element"""

    n = 8  # number of nodes
    r_n = 3  # number of natural basis parameters

    feb_name = "hex8"

    node_connectivity = [
        [1, 3, 4],  # 0
        [0, 2, 3],  # 1
        [1, 3, 6],  # 2
        [0, 2, 7],  # 3
        [0, 5, 7],  # 4
        [1, 4, 6],  # 5
        [2, 5, 7],  # 6
        [3, 4, 6],
    ]  # 7

    # Oriented positive = out
    face_nodes = (
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (4, 5, 6, 7),
        (0, 3, 2, 1),
    )

    # Vertex point locations in natural coordinates
    vloc = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    )

    # Gauss point locations
    g = 1.0 / 3.0**0.5
    gloc = (
        (-g, -g, -g),
        (g, -g, -g),
        (g, g, -g),
        (-g, g, -g),
        (-g, -g, g),
        (g, -g, g),
        (g, g, g),
        (-g, g, g),
    )

    # Guass weights
    gwt = (1, 1, 1, 1, 1, 1, 1, 1)

    @staticmethod
    def N(r, s, t):
        """Shape functions.

        8-element vector

        """
        n = [0.0] * 8
        n[0] = 0.125 * (1 - r) * (1 - s) * (1 - t)
        n[1] = 0.125 * (1 + r) * (1 - s) * (1 - t)
        n[2] = 0.125 * (1 + r) * (1 + s) * (1 - t)
        n[3] = 0.125 * (1 - r) * (1 + s) * (1 - t)
        n[4] = 0.125 * (1 - r) * (1 - s) * (1 + t)
        n[5] = 0.125 * (1 + r) * (1 - s) * (1 + t)
        n[6] = 0.125 * (1 + r) * (1 + s) * (1 + t)
        n[7] = 0.125 * (1 - r) * (1 + s) * (1 + t)
        return n

    @staticmethod
    def dN(r, s, t):
        """Shape function 1st derivatives."""
        dn = [np.zeros(3) for i in range(8)]
        # dN/dr
        dn[0][0] = -0.125 * (1 - s) * (1 - t)
        dn[1][0] = 0.125 * (1 - s) * (1 - t)
        dn[2][0] = 0.125 * (1 + s) * (1 - t)
        dn[3][0] = -0.125 * (1 + s) * (1 - t)
        dn[4][0] = -0.125 * (1 - s) * (1 + t)
        dn[5][0] = 0.125 * (1 - s) * (1 + t)
        dn[6][0] = 0.125 * (1 + s) * (1 + t)
        dn[7][0] = -0.125 * (1 + s) * (1 + t)
        # dN/ds
        dn[0][1] = -0.125 * (1 - r) * (1 - t)
        dn[1][1] = -0.125 * (1 + r) * (1 - t)
        dn[2][1] = 0.125 * (1 + r) * (1 - t)
        dn[3][1] = 0.125 * (1 - r) * (1 - t)
        dn[4][1] = -0.125 * (1 - r) * (1 + t)
        dn[5][1] = -0.125 * (1 + r) * (1 + t)
        dn[6][1] = 0.125 * (1 + r) * (1 + t)
        dn[7][1] = 0.125 * (1 - r) * (1 + t)
        # dN/dt
        dn[0][2] = -0.125 * (1 - r) * (1 - s)
        dn[1][2] = -0.125 * (1 + r) * (1 - s)
        dn[2][2] = -0.125 * (1 + r) * (1 + s)
        dn[3][2] = -0.125 * (1 - r) * (1 + s)
        dn[4][2] = 0.125 * (1 - r) * (1 - s)
        dn[5][2] = 0.125 * (1 + r) * (1 - s)
        dn[6][2] = 0.125 * (1 + r) * (1 + s)
        dn[7][2] = 0.125 * (1 - r) * (1 + s)
        return dn

    @staticmethod
    def ddN(r, s, t):
        """ "Shape function 2nd derivatives."""
        ddn = np.zeros((8, 3, 3))
        # dN/dr^2 = 0
        # dN / drds
        ddn[0][0][1] = 0.125 * (1 - t)
        ddn[1][0][1] = -0.125 * (1 - t)
        ddn[2][0][1] = 0.125 * (1 - t)
        ddn[3][0][1] = -0.125 * (1 - t)
        ddn[4][0][1] = 0.125 * (1 + t)
        ddn[5][0][1] = -0.125 * (1 + t)
        ddn[6][0][1] = 0.125 * (1 + t)
        ddn[7][0][1] = -0.125 * (1 + t)
        # dN / drdt
        ddn[0][0][2] = 0.125 * (1 - s)
        ddn[1][0][2] = -0.125 * (1 - s)
        ddn[2][0][2] = -0.125 * (1 + s)
        ddn[3][0][2] = 0.125 * (1 + s)
        ddn[4][0][2] = -0.125 * (1 - s)
        ddn[5][0][2] = 0.125 * (1 - s)
        ddn[6][0][2] = 0.125 * (1 + s)
        ddn[7][0][2] = -0.125 * (1 + s)
        # dN / dsdr
        ddn[0][1][0] = 0.125 * (1 - t)
        ddn[1][1][0] = -0.125 * (1 - t)
        ddn[2][1][0] = 0.125 * (1 - t)
        ddn[3][1][0] = -0.125 * (1 - t)
        ddn[4][1][0] = 0.125 * (1 + t)
        ddn[5][1][0] = -0.125 * (1 + t)
        ddn[6][1][0] = 0.125 * (1 + t)
        ddn[7][1][0] = -0.125 * (1 + t)
        # dN / ds^2 = 0
        # dN / dsdt
        ddn[0][1][2] = 0.125 * (1 - r)
        ddn[1][1][2] = 0.125 * (1 + r)
        ddn[2][1][2] = -0.125 * (1 + r)
        ddn[3][1][2] = -0.125 * (1 - r)
        ddn[4][1][2] = -0.125 * (1 - r)
        ddn[5][1][2] = -0.125 * (1 + r)
        ddn[6][1][2] = 0.125 * (1 + r)
        ddn[7][1][2] = 0.125 * (1 - r)
        # dN / dtdr
        ddn[0][2][0] = 0.125 * (1 - s)
        ddn[1][2][0] = -0.125 * (1 - s)
        ddn[2][2][0] = -0.125 * (1 + s)
        ddn[3][2][0] = 0.125 * (1 + s)
        ddn[4][2][0] = -0.125 * (1 - s)
        ddn[5][2][0] = 0.125 * (1 - s)
        ddn[6][2][0] = 0.125 * (1 + s)
        ddn[7][2][0] = -0.125 * (1 + s)
        # dN / dtds
        ddn[0][2][1] = 0.125 * (1 - r)
        ddn[1][2][1] = 0.125 * (1 + r)
        ddn[2][2][1] = -0.125 * (1 + r)
        ddn[3][2][1] = -0.125 * (1 - r)
        ddn[4][2][1] = -0.125 * (1 - r)
        ddn[5][2][1] = -0.125 * (1 + r)
        ddn[6][2][1] = 0.125 * (1 + r)
        ddn[7][2][1] = 0.125 * (1 - r)
        # dN/ dt^2 = 0
        return ddn


class Hex20(Element3D):
    """20-node (quadratic) hexahedral element"""

    pass


class Hex27(Element3D):
    """27-node (quadratic) hexahedral element"""

    n = 27  # number of nodes
    r_n = 3  # number of natural basis parameters
    feb_name = "hex27"  # label in FEBio XML


class Penta6(Element3D):
    """Functions for penta6 linear elements."""

    n = 6  # number of nodes
    r_n = 3  # number of natural basis parameters

    feb_name = "penta6"

    node_connectivity = [
        [1, 2, 3],  # 0
        [0, 2, 4],  # 1
        [0, 1, 5],  # 2
        [0, 4, 5],  # 3
        [1, 3, 5],  # 4
        [2, 3, 4],
    ]  # 5

    # Oriented positive = out
    face_nodes = ((0, 1, 4, 3), (1, 2, 4, 5), (2, 0, 3, 5), (0, 2, 1), (3, 4, 5))

    # Vertex point locations in natural coordinates.
    # TODO: confirm these coordinates are correct
    # vloc = ((0.0, 0.0, -1.0),
    #         (1.0, 0.0, -1.0),
    #         (0.0, 1.0, -1.0),
    #         (0.0, 0.0,  1.0),
    #         (1.0, 0.0,  1.0),
    #         (0.0, 1.0,  1.0))

    # Gauss point locations
    g = 1.0 / 3.0**0.5
    gloc = (
        (1 / 6, 1 / 6, -g),
        (2 / 3, 1 / 6, -g),
        (1 / 6, 2 / 3, -g),
        (1 / 6, 1 / 6, g),
        (2 / 3, 1 / 6, g),
        (1 / 6, 2 / 3, g),
    )

    # Guass weights
    gwt = (1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6)

    @staticmethod
    def N(r, s, t):
        """Shape functions.

        8-element vector

        """
        n = [0.0] * 6
        n[0] = 0.5 * (1 - r - s) * (1 - t)
        n[1] = 0.5 * r * (1 - t)
        n[2] = 0.5 * s * (1 - t)
        n[3] = 0.5 * (1 - r - s) * (1 + t)
        n[4] = 0.5 * r * (1 + t)
        n[5] = 0.5 * s * (1 + t)
        return n

    @staticmethod
    def dN(r, s, t):
        """Shape function 1st derivatives."""
        dn = [np.zeros(3) for i in range(6)]
        # dN/dr
        dn[0][0] = -0.5 * (1 - t)
        dn[1][0] = 0.5 * (1 - t)
        dn[2][0] = 0.0
        dn[3][0] = -0.5 * (1 + t)
        dn[4][0] = 0.5 * (1 + t)
        dn[5][0] = 0.0
        # dN/ds
        dn[0][1] = -0.5 * (1 - t)
        dn[1][1] = 0.0
        dn[2][1] = 0.5 * (1 - t)
        dn[3][1] = -0.5 * (1 + t)
        dn[4][1] = 0.0
        dn[5][1] = 0.5 * (1 + t)
        # dN/dt
        dn[0][2] = 0.0
        dn[1][2] = -0.5 * r
        dn[2][2] = -0.5 * s
        dn[3][2] = 0.0
        dn[4][2] = 0.5 * r
        dn[5][2] = 0.5 * s
        return dn

    @staticmethod
    def ddN(r, s, t):
        """ "Shape function 2nd derivatives."""
        ddn = np.zeros((6, 3, 3))
        # dN/dr^2 = 0
        # dN / drds = 0
        # dN / drdt
        ddn[0][0][2] = 0.5
        ddn[1][0][2] = -0.5
        ddn[2][0][2] = 0.0
        ddn[3][0][2] = -0.5
        ddn[4][0][2] = 0.5
        ddn[5][0][2] = 0.0
        # dN / dsdr = 0
        # dN / ds^2 = 0
        # dN / dsdt
        ddn[0][1][2] = 0.5
        ddn[1][1][2] = 0.0
        ddn[2][1][2] = -0.5
        ddn[3][1][2] = -0.5
        ddn[4][1][2] = 0.0
        ddn[5][1][2] = 0.5
        # dN / dtdr
        ddn[0][2][0] = 0.0
        ddn[1][2][0] = -0.5
        ddn[2][2][0] = 0.0
        ddn[3][2][0] = 0.0
        ddn[4][2][0] = 0.5
        ddn[5][2][0] = 0.0
        # dN / dtds
        ddn[0][2][1] = 0.0
        ddn[1][2][1] = 0.0
        ddn[2][2][1] = -0.5
        ddn[3][2][1] = 0.0
        ddn[4][2][1] = 0.0
        ddn[5][2][1] = 0.5
        # dN/ dt^2 = 0
        return ddn


class Quad4(Element2D):
    """Shape functions for quad4 bilinear shell element.

    This definition uses Guass point integration.  FEBio also has a
    nodal integration-type quad4 element defined; I'm not exactly sure
    which is used in the solver.

    """

    n = 4
    r_n = 2  # number of natural basis parameters
    node_connectivity = [[1, 3], [0, 2], [1, 3], [2, 0]]

    feb_name = "quad4"

    face_nodes = [[0, 1, 2, 3]]

    edge_nodes = [[0, 1], [1, 2], [2, 3], [3, 0]]

    # vertex point locations in natural coordinates
    vloc = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))

    a = 1.0 / 3.0**0.5
    gloc = ((-a, -a), (a, -a), (a, a), (-a, a))  # Gauss point locations
    gwt = (1, 1, 1, 1)  # Gauss weights

    def __init__(self, *args, **kwargs):
        super(Quad4, self).__init__(*args, **kwargs)
        self.properties["thickness"] = (1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def N(r, s, t=None):
        """Shape functions."""
        n = [0.0] * 4
        n[0] = 0.25 * (1 - r) * (1 - s)
        n[1] = 0.25 * (1 + r) * (1 - s)
        n[2] = 0.25 * (1 + r) * (1 + s)
        n[3] = 0.25 * (1 - r) * (1 + s)
        return n

    @staticmethod
    def dN(r, s, t=None):
        """Shape function 1st derivatives."""
        dn = [np.zeros(2) for i in range(4)]
        # d/dr
        dn[0][0] = -0.25 * (1 - s)
        dn[1][0] = 0.25 * (1 - s)
        dn[2][0] = 0.25 * (1 + s)
        dn[3][0] = -0.25 * (1 + s)
        # d/ds
        dn[0][1] = -0.25 * (1 - r)
        dn[1][1] = -0.25 * (1 + r)
        dn[2][1] = 0.25 * (1 + r)
        dn[3][1] = 0.25 * (1 - r)
        return dn

    @staticmethod
    def ddN(r, s, t=None):
        """Shape function 2nd derivatives."""
        ddn = np.zeros((4, 2, 2))
        # dN / dr² = 0
        # dN / drds
        ddn[0][0][1] = 0.25
        ddn[1][0][1] = -0.25
        ddn[2][0][1] = 0.25
        ddn[3][0][1] = -0.25
        # dN / dsdr
        ddn[0][1][0] = 0.25
        ddn[1][1][0] = -0.25
        ddn[2][1][0] = 0.25
        ddn[3][1][0] = -0.25
        # dN / ds² = 0

        return ddn

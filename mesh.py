import numpy as np
import febtools as feb
import febtools.element
from febtools import XpltReader
from febtools.element import elem_obj
import xml.etree.ElementTree as ET

class Mesh:
    """Stores a mesh geometry."""

    node = []
    element = []

    def __init__(self, node=None, element=None):

        # Node list
        if node is None:
            self.node = []
        else:
            self.node = node

        # Element list
        if element and node:
            if element[0] is febtools.element.Element:
                self.element = element
            else:
                self.element = [elem_obj(nid, node, eid=i)
                                for i, nid in enumerate(element)]

    def readfeb(self, f):
        """Read .feb file geometry"""
        root = ET.parse(f).getroot()
        if root.tag != "febio_spec":
            raise Exception("Root node is not 'febio_spec': '" +
                            fpath + "' is not a valid .feb file.")
        self.node = [tuple([float(a) for a in b.text.split(",")])
                     for b in root.findall("./Geometry/Nodes/*")]
        element = [tuple([int(a) - 1 for a in b.text.split(",")])
                   for b in root.findall("./Geometry/Elements/*")]
        element = [elem_obj(nid, self.node, eid=i)
                   for i, nid in enumerate(element)]
        self.element = element

    def elemcentroid(self):
        """List of element centroids (reference coordinates)."""
        centroid = []
        for i in range(len(self.element)):
            x = [self.node[inode] for inode in self.element[i]]
            c = [sum(v) / len(v) for v in zip(*x)]
            yield c

    def elemcoord(self):
        """Generator for element coordinates."""
        for idx in self.element:
            yield tuple([self.node[i] for i in idx])

    def find_nearest_node(self, x, y, z=None):
        """Find node nearest (x, y, z)

        """
        if z is None:
            p = (x, y, 0)
        else:
            p = (x, y, z)
        d = np.array(self.node) - p
        d = np.sum(d**2., axis=1)**0.5
        idx = np.argmin(abs(d))
        return idx

    def elem_with_node(self, idx):
        """Return elements containing a node

        idx := node id

        """
        eid = set(ii for (ii, r) in enumerate(self.element)
                  if idx in r.inode)
        elements = [self.element[i] for i in eid]
        return set(elements)

    def conn_elem(self, elements):
        """Find elements connected to elements.

        """
        nodes = set([i for e in elements
                     for i in e.inode])
        elements = []
        for idx in nodes:
            elements = elements + list(self.elem_with_node(idx))
        return set(elements)


class MeshSolution(Mesh):
    """Analysis of a solution step"""

    node = []
    element = []
    data = {}
    reader = None
    material_index = []

    
    def __init__(self, f=None, step=-1, matl_map=None):
        if f is None:
            # This is a minimal instance for debugging.
            pass
        else:
            if isinstance(f, str):
                self.reader = XpltReader(f)
            elif isinstance(f, XpltReader):
                self.reader = f
            self.node, self.element = self.reader.mesh()
            self.data = self.reader.stepdata(step)
            self.material_index_xplt = self.reader.material()
            self.material_map = matl_map

    def assign_materials(self, matl_map):
        """Assign materials from integer codes.

        mat_map : a dictionary mapping integers to material classes

        """
        for i, e in enumerate(self.element):
            matname = matl_map[e.mat_id]
            self.element[i].material = feb.material.getclass(matname)

    def f(self, istep = -1, r = 0, s = 0, t = 0):
        """Generator for F tensors for each element.
        
        Global coordinates: x, y, z
        Natural coordinates: r, s, t
        Displacements (global): u, v, w
    
        """
        for e in self.element:
            u = np.array([self.data['displacement'][i]
                          for i in e.inode])
            # displacements are exported for each node
            du_dR = np.dot(u.T, e.dN(r, s, t))
            du_dX = np.dot(np.linalg.inv(e.j((r, s, t))), du_dR)
            f = du_dX + np.eye(3)
            yield f
            
    def s(self):
        """1st Piola-Kirchoff stress for each element.
        
        The stress is calculated at the center of each element by
        transforming FEBio's Cauchy stress output.
        """
        s = []
        for t, f in zip(self.data['stress'], self.f()):
            # 1/J F S transpose(F) = t
            J = np.linalg.det(f)
            x = np.linalg.solve(1 / J * f, t)
            s = np.linalg.solve(f, x.T).T
            yield s

    def t(self):
        """Cauchy stress tensor for each element.

        The Cauchy stress is read directly from the xplt file.
        """
        for v in self.data['stress']:
            yield v

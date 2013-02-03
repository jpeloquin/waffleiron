import xml.etree.ElementTree as ET

class Mesh:
    """Stores a mesh geometry."""

    nodes = []
    elements = []

    def __init__(self):
        # Leave mesh blank

    def readfeb(f):
        """Read .feb file geometry"""
        root = ET.parse(fpath).getroot()
        if root.tag != "febio_spec":
            raise Exception("Root node is not 'febio_spec': "
                            "not a valid .feb file.")
        self.nodes = [tuple([float(a) for a in b.text.split(",")])
                      for b in root.findall("./Geometry/Nodes/*")]
        self.elements = [tuple([int(a) for a in b.text.split(",")])
                         for b in root.findall("./Geometry/Elements/*")]

class MeshSolution(Mesh):
    """Stores a mesh and its FEA solution."""

    def __init__(self):
        # Set up blank Xplt

    def read(f):
        # Read data from .xplt file into Xplt
        fdata = open(f,'rb').read()
        
    

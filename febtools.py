import xml.etree.ElementTree as ET
import struct

class Mesh:
    """Stores a mesh geometry."""
    
    def __init__(self):
        # Leave mesh blank
        self.nodes = []
        self.elements = []

    def readfeb(self, f):
        """Read .feb file geometry"""
        root = ET.parse(f).getroot()
        if root.tag != "febio_spec":
            raise Exception("Root node is not 'febio_spec': '" +
                            fpath + "' is not a valid .feb file.")
        self.nodes = [tuple([float(a) for a in b.text.split(",")])
                      for b in root.findall("./Geometry/Nodes/*")]
        self.elements = [tuple([int(a) for a in b.text.split(",")])
                         for b in root.findall("./Geometry/Elements/*")]
        
class MeshSolution(Mesh):
    """Stores a mesh and its FEA solution."""

    def __init__(self):
        # Set up blank Xplt
        self.step = []
        
    def readxplt(f):
        # Read data from .xplt file into Xplt
        fdata = open(f,'rb').read()
        
class Xpltreader:
    """Assists in reading an FEBio xplt file."""
    id2tag = {
        'root':            16777216,  # 0x01000000
        'header':          16842752,  # 0x01010000
        'version':         16842753,  # 0x01010001
        'nodes':           16842754,  # 0x01010002
        'dictionary':      16908288,  # 0x01020000
        'dict_item':       16908289,  # 0x01020001
        'item_type':       16908290,  # 0x01020002
        'item_format':     16908291,  # 0x01020003
        'item_name':       16908292,  # 0x01020004
        'global_var':      16912384,  # 0x01021000
        'material_var':    16916480,  # 0x01022000
        'nodeset_var':     16920576,  # 0x01023000
        'domain_var':      16924672,  # 0x01024000
        'surface_var':     16928768,  # 0x01025000
        'materials':       16973824,  # 0x01030000
        'material':        16973825,  # 0x01030001
        'material_id':     16973826,  # 0x01030002
        'material_name':   16973827,  # 0x01030003
        'geometry':        17039360,  # 0x01040000
        'node_section':    17043456,  # 0x01041000
        'node_coords':     17043457,  # 0x01041001
        'domain_section':  17047552,  # 0x01042000
        'domain':          17047808,  # 0x01042100
        'domain_header':   17047809,  # 0x01042101
        'elem_type':       17047810,  # 0x01042102
        'mat_id':          17047811,  # 0x01042103
        'elements':        17047812,  # 0x01042104
        'element_list':    17048064,  # 0x01042200
        'element':         17048065,  # 0x01042201
        'surface_section': 17051648,  # 0x01043000
        'surface':         17051904,  # 0x01043100
        'surface_header':  17051905,  # 0x01043101
        'surface_id':      17051906,  # 0x01043102
        'faces':           17051907,  # 0x01043103
        'face_list':       17052160,  # 0x01043200
        'face':            17052161,  # 0x01043201
        'state_section':   33554432,  # 0x02000000
        'state_header':    33619968,  # 0x02010000
        'time':            33619970,  # 0x02010002
        'state_data':      33685504,  # 0x02020000
        'state_var':       33685505,  # 0x02020001
        'variable_id':     33685506,  # 0x02020002
        'variable_data':   33685507,  # 0x02020003
        'global_data':     33685760,  # 0x02020100
        'material_data':   33686016,  # 0x02020200
        'node_data':       33686272,  # 0x02020300
        'domain_data':     33686528,  # 0x02020400
        'surface_data':    33686784   # 0x02020500
        }
    tag2id = dict((v, k) for k, v in id2tag.items())

    def __init__(self, fpath):
        """Read xplt data from file"""
        f = open(fpath,'rb')
        try:
            # init
            self.fdata = f.read()
            self.cursor = 0
            self.loc = {}
            # check endianness
            self.endian = '<' # initial assumption
            s = struct.pack('<I', self._dword())
            if s == 'BEF\x00':
                self.endian = '<'
            elif s == '\x00FEB':
                self.endian = '>'
            else:
                raise Exception("The first 4 bytes of " + fpath + " "
                                "do not match the FEBio spec: "
                                "it is not a valid .feb file.")
            # parse the file
            self._parseblock(len(self.fdata))
        finally:
            f.close()

    def _parseblock(self, stop, path = None):
        print "\nCalled parseblock()."
        print "Cursor = " + str(self.cursor)
        print "Stop at: " + str(stop)
        print "Current path: " + str(path)
        if path is None:
            path = []
        while self.cursor < stop:
            blockstart = self.cursor
            w = self._dword()
            sz = self._dword()
            blockend = blockstart + 8  + sz
            if w in self.tag2id:
                # found block
                print "The dword is " + str(w) + ", " + str(self.tag2id[w])
                print "Block: ", str(blockstart), "--", str(blockend)
                print "Cursor = " + str(self.cursor)
                self._parseblock(blockend, path + [self.tag2id[w]])
            else:
                # found data
                print "Found data."
                print "Cursor = " + str(self.cursor)
                print "Path: ", str(path)
                key = '/'.join([str(a) for a in path])
                self.loc[key] = blockstart
                print "Logged ", str(blockstart), ' at key ', key
                self.cursor = stop

    def _dword(self):
        dword = struct.unpack(self.endian+'I',
                              self.fdata[self.cursor:self.cursor+4]
                              )[0]
        self.cursor = self.cursor + 4
        return dword

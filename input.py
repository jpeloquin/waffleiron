#! /usr/bin/env python2.7

import xml.etree.ElementTree as ET
import struct
import numpy as np
import hex8

def readlog(fpath):
    """Reads FEBio logfile as a list of the steps' data.

    The function returns a list of dictionaries, one per solution
    step.  The keys are the variable names (e.g. x, sxy, J,
    etc.). Each dictionary value is a list of variable values over all
    the nodes.  The indexing of this list is the same as in the
    original file.
    
    This function can be used for both node and element data.

    """
    try:
        f = open(fpath, 'rU')
        allsteps = []
        stepdata = None
        for line in f:
            if line[0:5] == '*Data':
                keys = line[8:].strip().split(';')
                if stepdata is not None:
                    allsteps.append(stepdata)
                stepdata = {}
            elif line[0] != '*':
                linedata = []
                for i, s in enumerate(line.strip().split(',')[1:]):
                    try:
                        v = float(s)
                    except ValueError as e:
                        v = float('nan')
                    linedata.append(v)
                for k, v in zip(keys, linedata):
                    stepdata.setdefault(k, []).append(v)
        allsteps += [stepdata] # append last step
    finally:
        f.close()
    return allsteps
    

def readfeb(fpath):
    tree = ET.parse(fpath)
    root = tree.getroot()
    if root.tag != "febio_spec":
        raise Exception("Root node is not 'febio_spec': "
                        "not a valid .xplt file.")
    nodes = [tuple([float(a) for a in b.text.split(",")])
             for b in root.findall("./Geometry/Nodes/*")]
    elements = [tuple([int(a) for a in b.text.split(",")])
                for b in root.findall("./Geometry/Elements/*")]
    return nodes, elements

class Mesh:
    """Stores a mesh geometry."""

    node = []
    element = []

    def __init__(self):
        # Leave mesh blank
        self.node = []
        self.element = []

    def readfeb(self, f):
        """Read .feb file geometry"""
        root = ET.parse(f).getroot()
        if root.tag != "febio_spec":
            raise Exception("Root node is not 'febio_spec': '" +
                            fpath + "' is not a valid .feb file.")
        self.node = [tuple([float(a) for a in b.text.split(",")])
                      for b in root.findall("./Geometry/Nodes/*")]
        self.element = [tuple([int(a) for a in b.text.split(",")])
                         for b in root.findall("./Geometry/Elements/*")]

    def elemcentroid(self):
        """Generator for element centroids."""
        for i in range(len(self.element)):
            x = [self.node[inode] for inode in self.element[i]]
            c = [sum(v) / len(v) for v in zip(*x)]
            yield tuple(c)

    def elemcoord(self):
        """Generator for element coordinates."""
        for idx in self.element:
            yield tuple([self.node[i] for i in idx])


class MeshSolution(Mesh):
    """Analysis of a solution step"""

    node = []
    element = []
    data = {}
    fpath = ''

    def __init__(self, f = None, istep = -1):
        if f is None:
            # This is a minal instance for debugging.
            pass
        else:
            if isinstance(f, str):
                self.fpath = f
                reader = Xpltreader(f)
            elif isinstance(f, Xpltreader):
                reader = f
                self.fpath = reader.fpath
            self.node, self.element = reader.mesh()
            self.fpath = reader.fpath
            self.data = reader.solution(istep)

    def f(self, istep = -1, r = 0, s = 0, t = 0):
        """Generator for F tensors for each element.
        
        Global coordinates: x, y, z
        Natural coordinates: r, s, t
        Displacements (global): u, v, w
        """
        for i in range(len(self.element)):
            neln = len(self.element[i])
            X = np.array([self.node[a] 
                         for a in self.element[i]])
            u = np.array([self.data['displacement'][a]
                 for a in self.element[i]])
            if neln == 8:
                dN_dR = hex8.dshpfun(*(r, s, t))
            J = np.dot(dN_dR, X)
            du_dR = np.dot(dN_dR, u)
            du_dX = np.dot(np.linalg.inv(J), du_dR)
            f = du_dX + np.eye(3)
            yield f.T
            
    def s(self):
        """1st Piola-Kirchoff stress for each element.
        
        The stress is calculated at the center of each element by
        transforming FEBio's Cauchy stress output.
        """
        for t, f in zip(self.data['stress'], self.f()):
            # 1/J F S transpose(F) = t
            J = np.linalg.det(f)
            x = np.linalg.solve(1 / J * f, t)
            yield np.linalg.solve(f, x.T).T
            
        
class Xpltreader:
    """Parses an FEBio xplt file.

    """
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
        'elements':        16982276,  # 0x01032104
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
    tag2elem_type = {
        0: 'hex8',
        1: 'penta6',
        2: 'tet4',
        3: 'quad4',
        4: 'tri3',
        5: 'truss2'
        }
    tag2item_type = {
        0: 'float',
        1: 'vec3f',
        2: 'mat3fs'
        }
    tag2item_format = {
        0: 'node',
        1: 'item',
        2: 'mult'
        }

    fpath = '' # file path
    fdata = ''
    cursor = 0 # bytes from start of file
    endian = '' # little-endian: '<'
                # big-endian:    '>'
    time = [] # solution time of each step

    def __init__(self, fpath):
        """Read xplt data from file"""
        self.fpath = fpath
        print('Reading ' + fpath)
        f = open(fpath,'rb')
        try:
            self.fdata = f.read()
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
        finally:
            f.close()
        self.time = [struct.unpack(self.endian + 'f', s)[0] for s in
                self._lblock('state_section/state_header/time')]
        self.nsteps = len(self.time)

    def mesh(self):
        """Reads node and element lists"""
        data = self._lblock('root/geometry/'
                          'domain_section/domain/'
                          'element_list/element')
        element = []
        for s in data:
            v = struct.unpack('I' * (len(s) / 4), s)
            element.append(v[1:])
        s = self._lblock('root/geometry/'
                         'node_section/node_coords')[0]
        node = []
        for i in range(0, len(s), 12):
            node.append(struct.unpack('f' * 3, s[i:i+12]))
        return node, element

    def solution(self, istep):
        """Retrieve data for step istep

        The solution data is returned as a dictionary.  The data names
        (e.g. stress, displacement) are the keys.  These keys are read
        from the file's dictionary section.  Data is formatted into a
        list of floats, vectors, or tensors according to the data type
        specified in the file's dictionary section.

        """
        var = {}
        var['node'] = self._rdict('nodeset')
        var['domain'] = self._rdict('domain')
        var['surface'] = self._rdict('surface')
        nsteps = len(self.time)
        this_step = {}
        if istep == -1:
            istep = nsteps - 1
        for k, v in var.iteritems():
            if v:
                path = ('state_section/state_data/' + k + '_data'
                        '/state_var/variable_data')
                data = self._lblock(path)
                for i in range(0, len(v)):
                    s = data[istep * len(v) + i]
                    typ = v[i][0]
                    fmt = v[i][1]
                    name = v[i][2]
                    print('Found data named "' + name + '" in '
                          + k + ' section, '
                          'step ' + str(istep) + '.')
                    this_step[name] = self._unpackblock(s, typ)
        this_step['time'] = self.time[istep]
        return this_step

    def _rdict(self, name):
        """Find a dictionary section and read it.

        name = name of dictionary section to read: 'global',
        'material', 'nodeset', 'domain', or 'surface'.

        Returns a list of tuples: (type, format, name)

        type = 'float', 'vec3f' or 'mat3fs'
        format = 'node', 'item', or 'mult'
        name = a textual description of the data

        """
        dict = 'root/dictionary/' + name + '_var/dict_item'
        def trim(s): 
            return s[0:s.find('\x00')]
        if self._lblock(dict + '/item_type'):
            typ = [self.tag2item_type[self._dword(s)] for s 
                    in self._lblock(dict + '/item_type')]
            fmt = [self.tag2item_format[self._dword(s)] for s 
                      in self._lblock(dict + '/item_format')]
            name = [trim(s) for s 
                    in self._lblock(dict + '/item_name')]
            return zip(typ, fmt, name)
        else:
            return None

    def _unpackblock(self, s, typ):
        """Unpack binary data into floats, vectors, or matrices.

        s = the data

        type = the data type: 'float', 'vec3f' (vector), or
        'mat3fs' (matrix)

        Returns a tuple.

        """
        if len(s) == 0:
            raise Exception('Input data has zero length.')
        s = s[8:] # discard the 8 junk bytes at the start
        if typ == 'float':
            v = struct.unpack(self.endian + 'f' * (len(s) / 4), s)
        elif typ == 'vec3f':
            if len(s) % 12 != 0:
                raise Exception('Input data cannot be evenly divided '
                                'into vectors.')
            v = []
            for i in range(0, len(s), 12):
                v.append(np.array(
                        struct.unpack(self.endian + 'f' * 3,
                                      s[i:i+12])))
        elif typ == 'mat3fs':
            v = []
            if len(s) % 24 != 0:
                raise Exception('Input data cannot be evenly divided '
                                'into tensors.')
            for i in range(0, len(s), 24):
                a = struct.unpack(self.endian + 'f' * 6, s[i:i+24])
                # The FEBio database spec does not document the
                # tensor order, but this is correct (for now).
                v.append(np.array([[a[0], a[3], a[5]],
                                   [a[3], a[1], a[4]],
                                   [a[5], a[4], a[2]]]))
        else:
            raise Exception('Type ' + typ + ' not recognized.')
        return tuple(v)

    def _lblock(self, pathstr, start = 4, end = None):
        """List data from block(s).

        `_lblock()` searches fdata from the byte at `start` to the
        byte at `end` for all blocks matching the provided path and
        returns a flattened list of the data contained in those
        blocks.  Use caution if you expect multiple matches on
        multiple levels; this shouldn't happen in an FEBio file, but
        the spec doesn't forbid it.

        `pathstr` is a `/`-delimited sequence of block IDs. For
        example, to obtain the nodeset data dictiory, call

        `block('root/dictionary/nodeset_var')`

        ID names are given in the table at the end of the FEBio binary
        database specification.
        
        """
        blockpath = pathstr.split('/')
        if end is None:
            end = len(self.fdata) - 8
        result = []
        self.cursor = start
        while self.cursor < end:
            name, data = self._readblock()
            if name == blockpath[0]:
                s = self.cursor + 8
                e = s + len(data)
                if len(blockpath) > 1:
                    data = self._lblock('/'.join(blockpath[1:]), s, e)
                else:
                    data = [data]
                result = result + data
                self.cursor = e
            else:
                self.cursor = self.cursor + 8 + len(data)
        return result

    def _fblock(self, pathstr, start = 4, end = None):
        """Returns data from first occurrence of block."""
        block = self._lblock(pathstr, start, end)
        if len(block) > 1:
            raise Exception('Found multiple matches to ' 
                            + pathstr + '; only expected one.')
        if block:
            return block[0]
        else:
            return None

    def _readblock(self):
        """Reads a block starting at the current cursor location."""
        name, size = struct.unpack(self.endian + 'II',
                                   self.fdata[self.cursor:self.cursor+8])
        if self.cursor > len(self.fdata) - 8:
            raise Exception('The cursor is within 8 bytes of the end of the '
                            'file. There cannot be a valid block at this '
                            'position.')
        try:
            name = self.tag2id[name]
        except KeyError as e:
            print('_readblock did not find an expected tag at '
                  'cursor position ' + str(self.cursor))
        data = self.fdata[self.cursor+8:self.cursor+8+size]
        return name, data

    def _mvcursor(self, v):
        """Moves the cursor the specified number of bytes."""
        self.cursor = self.cursor + v
        if self.cursor < 0:
            raise Exception('Tried to move cursor before beginning of file.')
        elif self.cursor > len(self.fdata):
            raise Exception('Tried to move cursor after end of file.')

    def _dword(self, s = None):
        if s:
            dword = struct.unpack(self.endian + 'I', s)[0]
        else:
            dword = struct.unpack(self.endian+'I',
                                  self.fdata[self.cursor:self.cursor+4]
                                  )[0]
            self.cursor = self.cursor + 4
        return dword

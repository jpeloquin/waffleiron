#! /usr/bin/env python2.7

import xml.etree.ElementTree as ET
import struct
import numpy as np
import os
import febtools.element

def nstrip(string):
    """Remove trailing nulls from string.

    """
    for i, c in enumerate(string):
        if c == '\x00':
            return string[:i]
    return string


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


class XpltReader:
    """Parses an FEBio xplt file.

    """
    str2tag = {
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
    tag2str = dict((v, k) for k, v in str2tag.items())
    tag2elem_type = {
        0: febtools.element.Hex8,
        1: 'penta6',
        2: 'tet4',
        3: febtools.element.Quad4,
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

    fhandle = None
    endian = '' # little-endian: '<'
                # big-endian:    '>'
    time = []
    
    def __init__(self, fpath):
        """Read xplt data from file"""
        print('Reading ' + fpath)
        with open(fpath,'rb') as f:
            self.f = f
            self.endian = '<' # initial assumption
            s = f.read(4)
            if s == 'BEF\x00':
                self.endian = '<'
            elif s == '\x00FEB':
                self.endian = '>'
            else:
                raise Exception("The first 4 bytes of %s "
                                "do not match the FEBio spec: "
                                "it is not a valid .feb file."
                                % (fpath,))
            time = []
            a = self._findall('state_section')
            self.steploc = [loc for loc, sz in a]
            for l in self.steploc:
                a = self._findall('state_header/time', l)
                self.f.seek(a[0][0])
                s = self.f.read(a[0][1])
                time.append(struct.unpack(self.endian + 'f', s))
            self.time = time
    
    def mesh(self):
        """Reads node and element lists.

        Although FEBio splits elements into domains, here they are all
        concatenated into one list.  The domain classification in
        FEBio seems to map 1:1 to the material name anyway.

        """
        if self.f.closed:
            self.f = open(self.f.name, 'rb')
        try:
            # Read nodes
            node_list = []
            a = self._findall('root/geometry/node_section/'
                              'node_coords')
            for loc, sz in a:
                self.f.seek(loc)
                v = struct.unpack('f' * (sz / 4), self.f.read(sz))
                for i in xrange(0, len(v), 3):
                    node_list.append(tuple(v[i:i+3]))

            element_list = []
            domains =  self._findall('root/geometry/domain_section/domain')
            for loc, sz in domains:
                # Determine element type
                l, s = self._findall('domain_header/elem_type', loc)[0]
                self.f.seek(l)
                ecode = struct.unpack(self.endian
                                      + 'I',
                                      self.f.read(s))[0]
                etype = self.tag2elem_type[ecode]
                # Determine material id
                l, s = self._findall('domain_header/mat_id', loc)[0]
                self.f.seek(l)
                mat_id = struct.unpack(self.endian
                                       + 'I',
                                       self.f.read(s))[0]
                # Read elements
                elements = self._findall('element_list/element', loc)
                for l, s in elements:
                    self.f.seek(l)
                    data = self.f.read(s)
                    elem_id = struct.unpack(self.endian
                                            + 'I',
                                            data[0:4])[0]
                    elem_id = elem_id - 1 # 0-index
                    node_id = struct.unpack(self.endian 
                                          + 'I' * ((s - 1) / 4),
                                          data[4:])
                    # the nodes are already 0-indexed in the binary
                    # database
                    element = etype(node_id, node_list,
                                    elem_id=elem_id,
                                    matl_id=mat_id)
                    element_list.append(element)
        finally:
            self.f.close()
        return node_list, element_list

    def material(self):
        """Read material codes (integer -> name)

        """
        matl_index = []
        matloc = self._findall('root/materials/material')
        for loc, sz in matloc:
            st, sz = self._findall('material_id', loc)[0]
            if not sz == 4:
                raise Exception('Expected 4 byte integer as material id; '
                                'found {} byte sequence.'.format(sz))
            self.f.seek(st)
            mat_id = struct.unpack(self.endian + 'i', self.f.read(sz))[0]
            st, sz = self._findall('material_name', loc)[0]
            self.f.seek(st)
            mat_name = nstrip(self.f.read(sz))
            matl_index.append({'material_id': mat_id,
                              'material_name': mat_name})
        return matl_index

    def solution(self, step):
        """Retrieve data for step (1 indexed).

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
        if step == -1:
            istep = nsteps - 1
        else:
            istep = step - 1
        this_step = {}
        steploc = self.steploc[istep]
        for k, v in var.iteritems():
            if v:
                path = ('state_data/' + k + '_data'
                        '/state_var/variable_data')
                a = self._findall(path, steploc)
                for (loc, sz), (typ, fmt, name) in zip(a, v):
                    self.f.seek(loc)
                    s = self.f.read(sz)
                    print('Found data named %s in  %s section'
                          ', step %s.' % (name, k, str(istep)))
                    this_step[name] = \
                        self._unpack_variable_data(s, typ)
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
        path = 'root/dictionary/' + name + '_var/dict_item'
        a = self._findall(path)
        typ = []
        fmt = []
        name = []
        for loc, sz in a:
            for label, data in self._children(loc - 8):
                if label == 'item_type':
                    typ.append(self.tag2item_type[
                        struct.unpack(self.endian + 'I', data)[0]])
                elif label == 'item_format':
                    fmt.append(self.tag2item_format[
                        struct.unpack(self.endian + 'I', data)[0]])
                elif label == 'item_name':
                    name.append(nstrip(data))
                else:
                    raise Exception('%s block not expected as '
                                    'child of dict_item.' % (label,))
        return zip(typ, fmt, name)

    def _unpack_variable_data(self, s, typ):
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
            raise Exception('Type %s  not recognized.' % (str(typ),))
        return tuple(v)

    def _findall(self, pathstr, start = 0):
        """Finds position and size of blocks.
        
        self._findall(pathstr[, start])

        `pathstr` is a `/`-delimited sequence of block IDs. For
        example, to obtain the nodeset data dictiory, use
        `root/dictionary/nodeset_var`.

        `start` is where `_findall` starts searching.  Make sure it
        matches up with the start of a block _data section_.  For
        example, start at bit 12 to search only within root data.  The
        default is to search the entire file (bit 4 to end).

        The function returns a list of tuples (start, length)
        specifying, for each matching block, the start bit of the data
        section and its size in bits.  All possible matches are
        returned, considering each level of the search path.  Multiple
        matches are listed in the same order they appear in the file.

        """
        blockpath = pathstr.split('/')
        out = []
        if self.f.closed:
            self.f = open(self.f.name, 'rb')
        self.f.seek(start)
        # Look for block(s).
        if start == 0 or start == 4:
            end = os.path.getsize(self.f.name)
            self.f.seek(4)
        else:
            self.f.seek(-8, 1)
            label, size = self._bprops()
            end = self.f.tell() + size
        tlabel = blockpath[0]
        while self.f.tell() < end:
            label, size = self._bprops()
            if label == tlabel:
                if blockpath[1:]:
                    a = self._findall('/'.join(blockpath[1:]),
                                      self.f.tell())
                    out += a
                else:
                    loc = self.f.tell()
                    out += [(loc, size)]
                    self.f.seek(size, 1)
            else:
                 self.f.seek(size, 1)
        return out

    def _children(self, start):
        """Generator for children of a block.

        """
        if self.f.closed:
            self.f = open(self.f.name, 'rb')
        self.f.seek(start)
        label, size = self._bprops()
        end = self.f.tell() + size
        while self.f.tell() < end:
            label, size = self._bprops()
            data = self.f.read(size)
            yield label, data

    def _readblock(self):
        """Reads a block starting at the current cursor location.

        """
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

    def _dword(self):
        """Reads a 4-bit integer from the file.

        """
        s = self.f.read(4)
        if s:
            dword = struct.unpack(self.endian + 'I', s)[0]
        return dword

    def _bprops(self):
        """Reads label and size of a block.
        
        Returns (label, size) based on the next 2 dwords in the file.
        Label is in its string form and size is an integer.

        If less than 8 bits (2 dwords) remain in the file, returns
        None.

        """
        s = self.f.read(8)
        if len(s) == 8:
            d = struct.unpack(self.endian + 'II', s)
            return self.tag2str[d[0]], d[1]
        else:
            return None


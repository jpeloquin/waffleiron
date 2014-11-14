import warnings
import os

from lxml import etree as ET
import struct
import numpy as np
import febtools as feb
import febtools.element
from febtools.element import elem_obj
from operator import itemgetter

def _nstrip(string):
    """Remove trailing nulls from string.

    """
    for i, c in enumerate(string):
        if c == '\x00':
            return string[:i]
    return string

def load_model(fpath):
    """Loads a model (feb) and its solution (xplt).

    """
    if (fpath[-4:].lower() == '.feb'
        or fpath[-5:].lower() == '.xplt'):
        base, ext = os.path.splitext(fpath)
    else:
        base = fpath
    fp_feb = base + '.feb'
    fp_xplt = base + '.xplt'
    model = feb.input.FebReader(base + '.feb').model()
    if os.path.exists(fp_xplt):
        soln = feb.input.XpltReader(fp_xplt)
        model.apply_solution(soln)
    return model

def readlog(fpath):
    """Reads FEBio logfile as a list of the steps' data.

    The function returns a list of dictionaries, one per solution
    step.  The keys are the variable names (e.g. x, sxy, J,
    etc.). Each dictionary value is a list of variable values over all
    the nodes.  The indexing of this list is the same as in the
    original file.

    This function can be used for both node and element data.

    """
    f = open(fpath, 'rU')
    try:
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

class FebReader:
    """Read an FEBio xml file.

    """
    def __init__(self, file):
        """Read a file object as an FEBio xml file.

        """
        self.file = file
        self.root = ET.parse(self.file).getroot()
        if self.root.tag != "febio_spec":
            raise Exception("Root node is not 'febio_spec': '" +
                            fpath + "is not an FEBio xml file.")
        if self.root.attrib['version'] != '2.0':
            raise ValueError('{} is not febio_spec 2.0'.format(file))

    def materials(self):
        """Return dictionary of material objects keyed by id.

        """
        materials = {}
        for mat in self.root.findall('./Material/material'):
            # Read material attributes
            mat_type = mat.attrib['type']
            if 'name' in mat.attrib:
                name = mat.attrib['name']
            else:
                name = ''
            mat_id = int(mat.attrib['id']) - 1 # convert to 0-index
            if mat_type == "solid mixture":
                # collect child materials
                solids = []
                for child in mat:
                    solid = self._elem2mat(child)
                    solids.append(solid)
                material = febtools.material.SolidMixture(solids)
            else:
                material = self._elem2mat(mat)
            # Store material object in materials dictionary
            material.name = name
            materials[mat_id] = material
        return materials

    def _elem2mat(self, element):
        """Get material properties dictionary from a material element.

        """
        props = {}
        for child in element:
            v = map(float, child.text.split(','))
            if len(v) == 1:
                v = v[0]
            props[child.tag] = v
        cls = febtools.material.class_from_name[element.attrib['type']]
        return cls(props)

    def model(self):
        """Return model.

        """
        # Create model from mesh
        mesh = self.mesh() # materials handled in here
        model = feb.Model(mesh)
        # Boundary condition: fixed nodes
        for e_fix in self.root.findall("Boundary/fix"):
            lbl = e_fix.attrib['bc']
            node_ids = set()
            for e_node in e_fix:
                node_ids.add(int(e_node.attrib['id']))
            model.fixed_nodes[lbl].update(node_ids)
        # Load curves (sequences)
        # Steps
        for e_step in self.root.findall('Step'):
            e_module = e_step.find('Module')
            module = e_module.attrib['type']
            e_control = e_step.find('Control')
            e_boundary = e_step.find('Boundary')
            # TBD
        return model

    def mesh(self):
        """Return mesh.

        """
        materials = self.materials()
        nodes = [tuple([float(a) for a in b.text.split(",")])
                 for b in self.root.findall("./Geometry/Nodes/*")]
        # Read elements
        elements = [] # nodal index format
        for elset in self.root.findall("./Geometry/Elements"):
            mat_id = int(elset.attrib['mat']) - 1 # zero-index

            # map element type strings to classes
            cls = self._element_class(elset.attrib['type'])

            for elem in elset.findall("./elem"):
                ids = [int(a) - 1 for a in elem.text.split(",")]
                e = cls.from_ids(ids, nodes,
                                 material=materials[mat_id])
                elements.append(e)
        # Create mesh
        mesh = feb.Mesh(nodes, elements)
        return mesh

    @staticmethod
    def _element_class(label):
        """Return element class corresponding to type label.

        """
        d = {'quad4': feb.element.Quad4,
             'tri3': feb.element.Tri3,
             'hex8': feb.element.Hex8}
        return d[label]

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
        'domain_id':       16982277,  # 0x01032105
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
        4: febtools.element.Tri3,
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

    def __init__(self, fpath):
        """Load an .xplt file.

        """
        with open(fpath,'rb') as f:
            self.f = f

            # Endianness
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

            # Find timepoints
            time = []
            a = self._findall('state_section')
            self.steploc = [loc for loc, sz in a]
            for l in self.steploc:
                a = self._findall('state_header/time', l)
                self.f.seek(a[0][0])
                s = self.f.read(a[0][1])
                time.append(struct.unpack(self.endian + 'f', s)[0])
            self.times = time

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
                # convert 1-index to 0-index
                l, s = self._findall('domain_header/mat_id', loc)[0]
                self.f.seek(l)
                mat_id = struct.unpack(self.endian
                                       + 'I',
                                       self.f.read(s))[0] - 1
                # Read elements
                elements = self._findall('element_list/element', loc)
                for l, s in elements:
                    self.f.seek(l)
                    data = self.f.read(s)
                    elem_id = struct.unpack(self.endian
                                            + 'I',
                                            data[0:4])[0]
                    elem_id = elem_id - 1 # 0-index
                    node_ids = struct.unpack(self.endian
                                          + 'I' * ((s - 1) / 4),
                                          data[4:])
                    # the nodes are already 0-indexed in the binary
                    # database
                    element = etype.from_ids(node_ids, node_list,
                                             material=mat_id)
                    element_list.append(element)
        finally:
            self.f.close()
        node_list = np.array(node_list)
        mesh = feb.Mesh(node_list, element_list)
        return mesh

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
            mat_name = _nstrip(self.f.read(sz))
            matl_index.append({'material_id': mat_id,
                              'material_name': mat_name})
        return matl_index

    def step_index(self, time):
        """Return step index for a given time.

        """
        idx, d = min(enumerate(abs(t - time) for t in self.times),
                     key=itemgetter(1))
        return idx

    def stepdata(self, step=None, time=None):
        """Retrieve data for a specific solution step.

        The solution data is returned as a dictionary.  The data names
        (e.g. stress, displacement) are the keys.  These keys are read
        from the file's dictionary section.  Data is formatted into a
        list of floats, vectors, or tensors according to the data type
        specified in the file's dictionary section.

        The last step is the default.

        """
        if time is not None and step is None:
            step = self.step_index(time)
        elif time is None and step is None:
            step = -1
        elif time is not None and step is not None:
            raise Exception("Do not specify both `step` and `time`.")

        var = {}
        var['global'] = self._rdict('global')
        var['material'] = self._rdict('material')
        var['node'] = self._rdict('nodeset')
        var['domain'] = self._rdict('domain')
        var['surface'] = self._rdict('surface')

        data = {}
        data['time'] = self.times[step]

        steploc = self.steploc[step]
        for k, v in var.iteritems():
            if v:
                path = ('state_data/' + k + '_data'
                        '/state_var/variable_data')
                a = self._findall(path, steploc)
                for (loc, sz), (typ, fmt, name) in zip(a, v):
                    if sz == 0:
                        warnings.warn("{} data ({}, {}) at position {} has size {}".format(name, typ, fmt, loc, sz))
                    else:
                        self.f.seek(loc)
                        s = self.f.read(sz)
                        data.setdefault(k, {})[name] = \
                            self._unpack_variable_data(s, typ)

        # "element" is alias for FEBio's "domain category
        data['element'] = data['domain']
        return data

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
                    name.append(_nstrip(data))
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
        len_tot = len(s)

        # Validity check
        if len_tot == 0:
            raise Exception('Input data has zero length.')
        if typ not in ['float', 'vec3f', 'mat3fs']:
            raise Exception('Type %s  not recognized.' % (str(typ),))

        values = [] # list of values

        # iterate over any pseudo-blocks (region id, size, data) that
        # may exist
        i = 0
        while i < len(s):
            # read the block
            id, sz = struct.unpack('II', s[i:i+8])
            data = s[i+8:i+8+sz]
            i = i + 8 + sz
            # unpack and append the values
            if typ == 'float':
                v = list(struct.unpack(self.endian + 'f'*(len(data)/4),
                                       data))
            elif typ == 'vec3f':
                if len(data) % 12 != 0:
                    raise Exception('Input data cannot be '
                                    'evenly divided '
                                    'into vectors.')
                v = []
                for j in range(0, len(data), 12):
                    v.append(np.array(
                            struct.unpack(self.endian + 'f' * 3,
                                          data[j:j+12])))
            elif typ == 'mat3fs':
                v = []
                if len(data) % 24 != 0:
                    raise Exception('Input data cannot be '
                                    'evenly divided '
                                    'into tensors.')
                for j in range(0, len(data), 24):
                    a = struct.unpack(self.endian + 'f' * 6,
                                      data[j:j+24])
                    # The FEBio database spec does not document the
                    # tensor order
                    v.append(np.array([[a[0], a[3], a[5]],
                                       [a[3], a[1], a[4]],
                                       [a[5], a[4], a[2]]]))
            values = values + v
        return tuple(values)

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

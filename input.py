import warnings
import os

from lxml import etree as ET
import struct
import numpy as np
import febtools as feb
import febtools.element
from febtools.exceptions import UnsupportedFormatError
from operator import itemgetter

from . import xplt
from .conditions import Sequence
from .febioxml import control_tagnames_from_febio

def _nstrip(string):
    """Remove trailing nulls from string.

    """
    for i, c in enumerate(string):
        if c == '\x00':
            return string[:i]
    return string


# map FEBio xml boundary condition labels to internal labels
label_bc = {'x': 'x1',
            'y': 'x2',
            'z': 'x3',
            'p': 'pressure'}


def load_model(fpath):
    """Loads a model (feb) and its solution (xplt).

    This has been tested the most with FEBio file specification 2.0.

    """
    if (fpath[-4:].lower() == '.feb' or fpath[-5:].lower() == '.xplt'):
        base, ext = os.path.splitext(fpath)
    else:
        base = fpath
    fp_feb = base + '.feb'
    fp_xplt = base + '.xplt'
    try:
        model = feb.input.FebReader(fp_feb).model()
    except UnsupportedFormatError as err:
        # The .feb file is some unsupported version
        msg = "{}.  Falling back to defining the model from the .xplt "
        "file alone.  Values given only in the .feb file will not be "
        "available.  Using FEBio file format 2.x is recommended."
        msg = msg.format(err.message)
        warnings.warn(msg)
        # Attempt to work around the problem
        soln = feb.input.XpltReader(fp_xplt)
        model = feb.Model(soln.mesh())
        model.apply_solution(soln)
        return model
    # Apply the solution to the model
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
        allsteps += [stepdata]  # append last step
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
        version = self.root.attrib['version']
        if self.root.tag != "febio_spec":
            raise Exception("Root node is not 'febio_spec': '" +
                            file.name + "is not an FEBio xml file.")
        if version != '2.0':
            msg = '{} is not febio_spec 2.0'.format(file)
            raise UnsupportedFormatError(msg, file, version)

    def materials(self):
        """Return dictionary of materials keyed by id.

        """
        mats = {}
        mat_names = {}
        for m in self.root.findall('./Material/material'):
            # Read material into dictionary
            material = self._read_material(m)
            mat_id = int(m.attrib['id']) - 1  # FEBio counts from 1

            # Convert material to class if possible
            def convert_mat(d):
                if d['material'] in feb.material.class_from_name:
                    cls = feb.material.class_from_name[d['material']]
                    if d['material'] == 'solid mixture':
                        constituents = []
                        for d_child in d['constituents']:
                            constituents.append(convert_mat(d_child))
                        return cls(constituents)
                    else:
                        return cls(d['properties'])
                else:
                    raise NotImplementedError

            try:
                material = convert_mat(material)
            except NotImplementedError:
                warnings.warn("Warning: Material type `{}` is not implemented "
                              "for post-processing.  It will be represented "
                              "as a dictionary of properties."
                              "".format(m.attrib['type']))

            # Store material in index
            mats[mat_id] = material
            mat_names[mat_id] = m.attrib['type']
            # TODO: Use material names
        return mats, mat_names

    def _read_material(self, tag):
        """Get material properties dictionary from <material>.

        tag := the XML <material> element.

        A material is represented as

        """
        m = {}
        m['material'] = tag.attrib['type']
        m['properties'] = {}
        constituents = []
        for child in tag:
            if child.tag in ['material', 'solid']:
                # Child element is a material
                constituents.append(self._read_material(child))
            else:
                # Child element is a property
                m['properties'][child.tag] = self._read_property(child)
        if constituents:
            m['constituents'] = constituents
        return m

    def _read_property(self, tag):
        """Read a permeability element."""
        p = {}
        p.update(tag.attrib)
        for child in tag:
            p_child = self._read_property(child)
            p[child.tag] = p_child
        v = tag.text.lstrip().rstrip()
        if v:
            v = [float(a) for a in v.split(',')]
            if len(v) == 1:
                v = v[0]
            if not p:
                return v
            else:
                p['value'] = v
        return p

    def model(self):
        """Return model.

        """
        # Create model from mesh
        mesh = self.mesh()
        model = feb.Model(mesh)
        # Store materials
        model.materials, model.material_names = self.materials()
        # Boundary condition: fixed nodes
        internal_label = {'x': 'x1',
                          'y': 'x2',
                          'z': 'x3',
                          'p': 'pressure'}
        # TODO: Solutes
        for e_fix in self.root.findall("Boundary/fix"):
            lbl = internal_label[e_fix.attrib['bc']]
            # Convert from FEBio XML label to internal label, if
            # conversion is provided
            if lbl in label_bc:
                lbl = label_bc[lbl]
            node_ids = set()
            for e_node in e_fix:
                node_ids.add(int(e_node.attrib['id']))
            model.fixed_nodes[lbl].update(node_ids)
        # Load curves (sequences)
        sequences = {}
        for e_lc in self.root.findall('LoadData/loadcurve'):
            def parse_pt(text):
                x, y = text.split(',')
                return float(x), float(y)
            curve = [parse_pt(a.text) for a in e_lc.getchildren()]
            if 'extend' in e_lc.attrib:
                extend = e_lc.attrib['extend']
            else:
                extend = 'extrapolate'  # default
            if 'type' in e_lc.attrib:
                typ = e_lc.attrib['type']
            else:
                typ = 'linear'  # default
            sequences[e_lc.attrib['id']] = Sequence(curve, typ=typ, extend=extend)
        # Steps
        model.steps = []
        for e_step in self.root.findall('Step'):
            step = {'control': {}}
            # Module
            e_module = e_step.find('Module')
            step['module'] = e_module.attrib['type']
            # Control section
            e_control = e_step.find('Control')
            for e in e_control:
                if e.tag in control_tagnames_from_febio:
                    # TODO: handle types correctly
                    step['control'][control_tagnames_from_febio[e.tag]] = e.text
            # Control/time_stepper section
            step['control']['time stepper'] = {}
            e_stepper = e_control.find('time_stepper')
            for e in e_stepper:
                if e.tag in control_tagnames_from_febio:
                    # TODO: handle types correctly
                    k = control_tagnames_from_febio[e.tag]
                    step['control']['time stepper'][k] = float(e.text)
            # Handle dtmax; it might have an associated sequence
            e_dtmax = e_stepper.find('dtmax')
            if 'lc' in e_dtmax.attrib:
                # dtmax has a must point sequence
                step['control']['time stepper']['dtmax'] = sequences[e_dtmax.attrib['lc']]
            model.steps.append(step)
        return model

    def mesh(self):
        """Return mesh.

        """
        mats, mat_names = self.materials()
        nodes = [tuple([float(a) for a in b.text.split(",")])
                 for b in self.root.findall("./Geometry/Nodes/*")]
        # Read elements
        elements = []  # nodal index format
        for elset in self.root.findall("./Geometry/Elements"):
            mat_id = int(elset.attrib['mat']) - 1  # zero-index

            # map element type strings to classes
            cls = self._element_class(elset.attrib['type'])

            for elem in elset.findall("./elem"):
                ids = [int(a) - 1 for a in elem.text.split(",")]
                e = cls.from_ids(ids, nodes,
                                 mat_id=mat_id,
                                 mat=mats[mat_id])
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
    def __init__(self, f):
        """Load an .xplt file.

        """
        def from_fobj(self, f):
            self.f = f

            # Endianness
            f.seek(0)
            self.endian = xplt.parse_endianness(f.read(4))

            # Find timepoints
            time = []
            a = self._findall('state')
            self.steploc = [loc for loc, sz in a]
            for l in self.steploc:
                a = self._findall('state header/time', l)
                self.f.seek(a[0][0])
                s = self.f.read(a[0][1])
                time.append(struct.unpack(self.endian + 'f', s)[0])
            self.times = time

            return self

        if type(f) is str:
            fpath = f
            with open(fpath, 'rb') as f:
                from_fobj(self, f)
        else:
            from_fobj(self, f)

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
                v = struct.unpack('f' * (sz // 4), self.f.read(sz))
                for i in range(0, len(v), 3):
                    node_list.append(tuple(v[i:i+3]))

            element_list = []
            domains = self._findall('root/geometry/domain_section/domain')
            for loc, sz in domains:
                # Determine element type
                l, s = self._findall('domain_header/elem_type', loc)[0]
                self.f.seek(l)
                ecode = struct.unpack(self.endian
                                      + 'I',
                                      self.f.read(s))[0]
                etype = xplt.element_type_from_id[ecode]
                if type(etype) is str:
                    msg = "`{}` element type is not implemented."
                    raise NotImplementedError(msg.format(etype))
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
                    elem_id = elem_id - 1  # 0-index
                    node_ids = struct.unpack(self.endian
                                             + 'I' * ((s - 1) // 4),
                                             data[4:])
                    # the nodes are already 0-indexed in the binary
                    # database
                    element = etype.from_ids(node_ids, node_list,
                                             mat_id=mat_id)
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
            b = self.f.read(sz)
            mat_name = b[:b.find(b'\x00')].decode()
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
        for k, v in var.items():
            if v:
                path = ('state data/' + k + ' data'
                        '/state variable/data')
                a = self._findall(path, steploc)
                for (loc, sz), (typ, fmt, name) in zip(a, v):
                    if sz == 0:
                        msg = "{} data ({}, {}) at position {} has size {}"
                        warnings.warn(msg.format(name, typ, fmt, loc, sz))
                    else:
                        self.f.seek(loc)
                        s = self.f.read(sz)
                        data.setdefault(k, {})[name] = \
                            self._unpack_variable_data(s, typ)

        # "element" is alias for FEBio's "domain" category
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
        path = 'root/dictionary/' + name + ' variables/dictionary item'
        a = self._findall(path)
        typ = []
        fmt = []
        name = []
        for loc, sz in a:
            for label, data in self._children(loc - 8):
                if label == 'item type':
                    typ.append(xplt.item_type_from_id[
                        struct.unpack(self.endian + 'I', data)[0]])
                elif label == 'item format':
                    fmt.append(xplt.item_format_from_id[
                        struct.unpack(self.endian + 'I', data)[0]])
                elif label == 'item name':
                    name.append(data[:data.find(b'\x00')].decode())
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

        values = []  # list of values

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
                fmt = self.endian + 'f' * int(len(data)/4)
                v = list(struct.unpack(fmt, data))
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

    def _findall(self, pathstr, start=0):
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
            return xplt.tags_table[d[0]]['name'], d[1]
        else:
            return None

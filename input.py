import warnings
import os

from lxml import etree as ET
import struct
import numpy as np
import febtools.element
from febtools.exceptions import UnsupportedFormatError
from operator import itemgetter

from .core import Model, Mesh, Body, ImplicitBody, Sequence, ScaledSequence, NodeSet, FaceSet, ElementSet, RigidInterface
from . import xplt
from . import febioxml, febioxml_2_5, febioxml_2_0
from . import material as material_lib
from .febioxml import control_tagnames_from_febio, elem_cls_from_feb

def _nstrip(string):
    """Remove trailing nulls from string.

    """
    for i, c in enumerate(string):
        if c == '\x00':
            return string[:i]
    return string


def _to_number(s):
    """Convert numeric string to int or float as appropraite."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def _maybe_to_number(s):
    """Convert string to number if possible, otherwise return string."""
    try:
        return _to_number(s)
    except ValueError:
        return s

def _read_parameter(e, sequence_dict):
    """Read a parameter from an XML element.

    The parameter may be fixed or variable.  If variable, a Sequence or
    ScaledSequence will be returned.

    """
    # Check if this is a time-varying or fixed property
    if "lc" in e.attrib:
        # The property is time-varying
        seq_id = int(e.attrib["lc"]) - 1
        sequence = sequence_dict[seq_id]
        if e.text is not None and e.text.strip() != "":
            scale = _to_number(e.text)
            return ScaledSequence(sequence, scale)
        else:
            return sequence
    else:
        # The property is fixed
        return _to_number(e.text)


def load_model(fpath):
    """Loads a model (feb) and the solution (xplt) if it exists.

    The following data is supported for FEBio XML 2.0 and 2.5:
    - Materials
    - Geometry: nodes and elements

    """
    if (fpath[-4:].lower() == '.feb' or fpath[-5:].lower() == '.xplt'):
        base, ext = os.path.splitext(fpath)
    else:
        base = fpath
    # Attempt to read the FEBio xml file
    fp_feb = base + '.feb'
    try:
        model = FebReader(fp_feb).model()
        feb_ok = True
    except UnsupportedFormatError as err:
        # The .feb file is some unsupported version
        msg = "{}.  Falling back to defining the model from the .xplt "
        "file alone.  Values given only in the .feb file will not be "
        "available.  Using FEBio file format 2.x is recommended."
        msg = msg.format(err.message)
        warnings.warn(msg)
        feb_ok = False
    # Attempt to read the xplt file, if it exists
    fp_xplt = base + '.xplt'
    if os.path.exists(fp_xplt):
        with open(fp_xplt, 'rb') as f:
            soln = xplt.XpltData(f.read())
        xplt_ok = True
    else:
        xplt_ok = False
    # Combine the feb and xplt data, preferring the feb data for model definition
    if feb_ok and xplt_ok:
        model.apply_solution(soln)
    elif not feb_ok and xplt_ok:
        # Use the xplt file to construct a model
        model = Model(soln.mesh())
        model.apply_solution(soln)
    elif not feb_ok and not xplt_ok:
        raise ValueError("Neither `{}` nor `{}` could be read.  Check that they exist "
                         "and are accessible.".format(fp_feb, fp_xplt))
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
        self.feb_version = self.root.attrib['version']
        if self.root.tag != "febio_spec":
            raise Exception("Root node is not 'febio_spec': '" +
                            file.name + "is not an FEBio xml file.")
        if self.feb_version not in ['2.0', '2.5']:
            msg = 'FEBio XML version {} is not supported by febtools'.format(self.feb_version)
            raise UnsupportedFormatError(msg, file, self.feb_version)
        self._sequences = None  # memo for sequences()

    def materials(self):
        """Return dictionary of materials keyed by id.

        """
        mats = {}
        mat_labels = {}
        for m in self.root.findall('./Material/material'):
            # Read material into dictionary
            material = self._read_material(m)
            mat_id = int(m.attrib['id']) - 1  # FEBio counts from 1
            try:
                material = mat_obj_from_elemd(material)
            except NotImplementedError:
                warnings.warn("Warning: Material type `{}` is not implemented "
                              "for post-processing.  It will be represented "
                              "as a dictionary of properties."
                              "".format(m.attrib['type']))

            # Store material in index
            mats[mat_id] = material
            if 'name' in m.attrib:
                mat_labels[mat_id] = m.attrib['name']
            else:
                mat_labels[mat_id] = str(mat_id)
        return mats, mat_labels


    @property
    def sequences(self):
        """Return dictionary of sequences (load curves) keyed by ID.

        Sequence IDs are integers starting from 0.

        The first time this method is called for a FebReader object it
        creates and memoizes a set of Sequence objects.  Subsequent
        calls return the same Sequence objects.

        """
        if self._sequences is None:
            self._sequences = {}
            for ord_id, e_lc in enumerate(self.root.findall('LoadData/loadcurve')):
                pseudo_id = int(e_lc.attrib['id'])
                def parse_pt(text):
                    x, y = text.split(',')
                    return float(x), float(y)
                curve = [parse_pt(a.text) for a in e_lc.getchildren()]
                # Set extrapolation
                if 'extend' in e_lc.attrib:
                    extend = e_lc.attrib['extend']
                else:
                    extend = 'extrapolate'  # default
                # Set interpolation
                if 'type' in e_lc.attrib:
                    typ = e_lc.attrib['type']
                else:
                    typ = 'linear'  # default
                # Create and store the Sequence object
                self._sequences[ord_id] = Sequence(curve, typ=typ, extend=extend)
        return self._sequences


    def _read_material(self, tag):
        """Get material properties dictionary from <material>.

        tag := the XML <material> element.

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
        """Read a material property element."""
        # Check if this is a time-varying or fixed property
        if "lc" in tag.attrib:
            # The property is time-varying
            seq_id = int(tag.attrib["lc"]) - 1
            sequence = self.sequences[seq_id]
            scale = _to_number(tag.text)
            return ScaledSequence(sequence, scale)
        else:
            # The property is fixed
            p = {}
            p.update(tag.attrib)
            for child in tag:
                p_child = self._read_property(child)
                p[child.tag] = p_child
            if tag.text is not None:
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
        # Get the materials dictionary so we can assign materials to
        # elements when we read the geometry
        materials_by_id, material_labels = self.materials()
        # Create model geoemtry
        mesh = self.mesh(materials=materials_by_id)
        model = Model(mesh)
        # Store the materials and their labels, now that the Model
        # object has been instantiated
        for ord_id, material in materials_by_id.items():
            name = material_labels[ord_id]
            model.named["materials"].add(name, material)
            model.named["materials"].add(ord_id, material, nametype="ordinal_id")
        # Read and store named sets of geometry
        named_sets = febioxml.read_named_sets(self.root)
        for entity_type in named_sets:
            for name in named_sets[entity_type]:
                obj = named_sets[entity_type][name]
                model.named[entity_type].add(name, obj)

        # Read explicit rigid bodies.  Create a Body object for each
        # rigid body "material" in the XML with explicit geometry.
        EXPLICIT_BODIES = {}
        for e_elements in self.root.findall("Geometry/Elements"):
            mat_id = int(e_elements.attrib["mat"]) - 1
            mat = model.named["materials"].obj(mat_id, nametype="ordinal_id")
            if isinstance(mat, material_lib.RigidBody):
                elements = []
                for e_elem in e_elements:
                    eid = int(e_elem.attrib["id"]) - 1
                    elements.append(model.mesh.elements[eid])
                EXPLICIT_BODIES[mat_id] = Body(elements)
        # Read implicit rigid bodies.
        IMPLICIT_BODIES = {}
        for e_impbod in self.root.findall("Boundary/rigid"):
            mat_id = int(e_impbod.attrib["rb"]) - 1
            mat = model.named["materials"].obj(mat_id, nametype="ordinal_id")
            node_ids = model.named["node sets"].obj(e_impbod.attrib["node_set"])
            IMPLICIT_BODIES[mat_id] = ImplicitBody(model.mesh, node_ids, mat)

        # Boundary condition: fixed nodes
        axis_name_conv_from_xml = {'x': 'x1',
                                   'y': 'x2',
                                   'z': 'x3',
                                   'p': 'fluid'}
        var_from_fix_tag_axis = {'x1': 'displacement',
                                 'x2': 'displacement',
                                 'x3': 'displacement',
                                 'fluid': 'pressure'}
        dof_name_conv_from_xml = {"x": "x1",
                                  "y": "x2",
                                  "z": "x3",
                                  "Rx": "α1",
                                  "Ry": "α2",
                                  "Rz": "α3"}
        # ^ Mapping of axis → constrained variable for <fix> tags in
        # FEBio XML.

        # Read fixed boundary conditions. TODO: Support solutes
        #
        # Read fixed constraints on node sets:
        for e_fix in self.root.findall("Boundary/fix"):
            # Each <fix> tag may specify multiple bc labels.  Split them
            # up and convert each to febtools naming convention.
            if self.feb_version == '2.0':
                # In FEBio XML 2.0, bc labels are concatenated.
                fixed = [axis_name_conv_from_xml[bc] for bc in
                         febioxml_2_0.split_bc_names(e_fix.attrib['bc'])]
            elif self.feb_version == '2.5':
                # In FEBio XML 2.5, bc labels are comma-delimeted.
                fixed = [axis_name_conv_from_xml[bc] for bc in
                         febioxml_2_5.split_bc_names(e_fix.attrib['bc'])]
            # For each axis, apply the fixed BCs to the model.
            for ax in fixed:
                # Get the nodeset that is constrained
                if self.feb_version == "2.0":
                    # In FEBio XML 2.0, each node to which the fixed boundary
                    # condition is applied is listed under the <fix> tag.
                    node_ids = NodeSet()
                    for e_node in e_fix:
                        node_ids.add(int(e_node.attrib['id']) - 1)
                elif self.feb_version == "2.5":
                    # In FEBio XML 2.5, the node set to which the fixed
                    # boundary condition is applied is referenced by name.
                    node_ids = model.named["node sets"].obj(e_fix.attrib["node_set"])
                var = var_from_fix_tag_axis[ax]
                # Hack to deal with febtools' model.fixed dict only
                # supporting fixed displacement and pressure for nodes.
                # At the time of this writing the axis / variable split
                # hasn't been implemented for fixed boundary conditions.
                if var == "pressure":
                    ax = "pressure"
                model.fixed['node'][ax].update(node_ids)
        #
        # Read fixed constraints on rigid bodies:
        for e_fix in self.root.findall("Boundary/rigid_body"):
            # Get the Body object to which <rigid_body> refers to by
            # material id.
            mat_id = int(e_fix.attrib["mat"]) - 1
            if mat_id in EXPLICIT_BODIES:
                body = EXPLICIT_BODIES[mat_id]
            else:
                # Assume mat_id is refers to an implicit rigid body
                body = IMPLICIT_BODIES[mat_id]
            for e_dof in e_fix.findall("fixed"):
                dof = dof_name_conv_from_xml[e_dof.attrib["bc"]]
                model.fixed["body"][dof].add(body)
        #
        # Read rigid body ↔ node set rigid interfaces
        for e_rigid in self.root.findall("Boundary/rigid"):
            body = model.named["materials"].obj(int(e_rigid.attrib["rb"]) - 1,
                                                nametype="ordinal_id")
            node_set = model.named["node sets"].obj(e_rigid.attrib["node_set"])
            rigid_interface = RigidInterface(body, node_set)
            model.constraints.append(rigid_interface)

        # Load curves (sequences)
        for seq_id, seq in self.sequences.items():
            model.named["sequences"].add(seq_id, seq, nametype="ordinal_id")

        # Steps
        model.steps = []
        for e_step in self.root.findall('Step'):
            step = {'control': {}}
            # Module
            e_module = e_step.find('Module')
            if e_module is not None:
                step['module'] = e_module.attrib['type']
            # Control section
            e_control = e_step.find('Control')
            for e in e_control:
                if e.tag in control_tagnames_from_febio:
                    step['control'][control_tagnames_from_febio[e.tag]] =\
                        _maybe_to_number(e.text)
            # Control/time_stepper section
            step['control']['time stepper'] = {}
            e_stepper = e_control.find('time_stepper')
            for e in e_stepper:
                if e.tag in control_tagnames_from_febio:
                    k = control_tagnames_from_febio[e.tag]
                    step['control']['time stepper'][k] = _read_parameter(e, self.sequences)
            model.steps.append(step)

        # Output variables
        output_variables = []
        for e_var in self.root.findall("Output/plotfile/var"):
            output_variables.append(e_var.attrib["type"])
        model.output["variables"] = output_variables

        return model

    def mesh(self, materials=None):
        """Return mesh.

        """
        if materials is None:
            materials, mat_labels = self.materials()
        nodes = [tuple([float(a) for a in b.text.split(",")])
                 for b in self.root.findall("./Geometry/Nodes/*")]
        # Read elements
        elements = []  # nodal index format
        for elset in self.root.findall("./Geometry/Elements"):
            mat_id = int(elset.attrib['mat']) - 1  # zero-index

            # map element type strings to classes
            cls = elem_cls_from_feb[elset.attrib['type']]

            for elem in elset.findall("./elem"):
                ids = [int(a) - 1 for a in elem.text.split(",")]
                e = cls.from_ids(ids, nodes,
                                 mat_id=mat_id,
                                 mat=materials[mat_id])
                elements.append(e)
        # Create mesh
        mesh = Mesh(nodes, elements)
        return mesh


def mat_obj_from_elemd(d):
    """Convert material element to material object"""
    if d["material"] in febioxml.solid_class_from_name:
        cls = febioxml.solid_class_from_name[d["material"]]
        if d["material"] == "solid mixture":
            constituents = []
            for d_child in d["constituents"]:
                constituents.append(mat_obj_from_elemd(d_child))
            return cls(constituents)
        elif d["material"] == "biphasic":
            # Instantiate Permeability object
            p_props = d["properties"]["permeability"]
            typ = d["properties"]["permeability"]["type"]
            p_class = febioxml.perm_class_from_name[typ]
            permeability = p_class.from_feb(**p_props)
            # Instantiate solid constituent object
            if len(d["constituents"]) > 1:
                # TODO: Specify which material in the error message.
                # This requires retaining material ids in the dict
                # passed to this function.
                raise ValueError("""A porelastic solid was encountered with {len(d['constituents'])} solid constituents.  Poroelastic solids must have exactly one solid constituent.""")
            solid = mat_obj_from_elemd(d["constituents"][0])
            # Return the Poroelastic Solid object
            return material_lib.PoroelasticSolid(solid, permeability)
        elif hasattr(cls, "from_feb") and callable(cls.from_feb):
            return cls.from_feb(**d["properties"])
        else:
            return cls(d["properties"])
    else:
        raise ValueError(f"{d['material']} is not supported in the loading of FEBio XML.")


class XpltReader:
    """Parses an FEBio xplt file.

    Obsolete.  Does not support FEBio > 2.5.

    """
    def __init__(self, f):
        """Load an .xplt file.

        """
        warnings.warn("XpltReader is deprected in favor of XpltData",
                      DeprecationWarning)

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
        mesh = Mesh(node_list, element_list)
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

    def step_data(self, step=None, time=None):
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
        var['global variables'] = self._rdict('global')
        var['material variables'] = self._rdict('material')
        var['node variables'] = self._rdict('node')
        var['element variables'] = self._rdict('element')
        var['surface variables'] = self._rdict('surface')

        data = {}
        data['time'] = self.times[step]

        steploc = self.steploc[step]
        for k, v in var.items():
            if v:
                path = ('state data/' + k.split(' ')[0] + ' data' +
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

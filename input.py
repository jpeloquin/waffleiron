import warnings
import os
from pathlib import Path

from lxml import etree as ET
import struct
import numpy as np
import pandas as pd

import febtools.element
from febtools.exceptions import UnsupportedFormatError
from operator import itemgetter

from .math import orthonormal_basis, vec_from_sph
from .model import Model, Mesh
from .core import Body, ImplicitBody, Sequence, ScaledSequence, NodeSet, FaceSet, ElementSet, RigidInterface
from . import xplt
from . import febioxml, febioxml_2_5, febioxml_2_0
from . import material as material_lib
from .febioxml import control_tagnames_from_febio, control_values_from_febio, elem_cls_from_feb, normalize_xml, _to_number, _maybe_to_number, _find_unique_tag, VAR_FROM_XML_NODE_BC, DOF_NAME_FROM_XML_NODE_BC

def _nstrip(string):
    """Remove trailing nulls from string.

    """
    for i, c in enumerate(string):
        if c == '\x00':
            return string[:i]
    return string


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


def _vec_from_text(s) -> tuple:
    return tuple(_to_number(x.strip()) for x in s.split(","))


def read_febio_xml(f):
    """Return lxml tree for FEBio XML file path or IO object."""
    parser = ET.XMLParser(remove_blank_text=True)
    try:
        f = open(f, "rb")
    except TypeError:
        # Assume pth is already an IO object; caller may have some reason to
        # hold it open, or just never wrote the file to disk.
        tree = ET.parse(f, parser)
    else:
        try:
            tree = ET.parse(f, parser)
        finally:
            f.close()
    return tree


def load_model(fpath):
    """Loads a model (feb) and the solution (xplt) if it exists.

    The following data is supported for FEBio XML 2.0 and 2.5:
    - Materials
    - Geometry: nodes and elements

    """
    if isinstance(fpath, str):
        fpath = Path(fpath)
    # Attempt to read the FEBio xml file
    try:
        model = FebReader(str(fpath.with_suffix(".feb"))).model()
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
    fp_xplt = fpath.with_suffix(".xplt")
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


def textdata_list(fpath, delim=" "):
    """Reads FEBio logfile as a list of the steps' data.

    The function returns a list of dictionaries, one per solution step.
    The keys are the variable names (e.g. x, sxy, J, etc.). Each
    dictionary value is a list of variable values over all the nodes.
    The indexing of this list is the same as in the original file.

    This function can be used for both node and element data.

    """
    with open(fpath, 'r') as f:
        steps = []
        # Find column names
        colnames = None
        for l in f:
            if l.startswith("*Data  = "):
                colnames = l.lstrip("*Data  = ").rstrip().split(";")
            elif not l.startswith("*"):
                break
        if colnames is None:
            raise ValueError(f"Could not find '*Data' row in {f.name}")
        # Read the data
        f.seek(0)
        stepdata = None
        for l in f:
            if l.startswith('*'):
                if stepdata is not None:
                    steps.append(stepdata)
                stepdata = None
            else:
                if stepdata is None:
                    stepdata = {}
                for k, s in zip(colnames, l.strip().split(delim)[1:]):
                    try:
                        v = float(s)
                    except ValueError as e:
                        v = float('nan')
                    stepdata.setdefault(k, []).append(v)
        steps.append(stepdata)  # add the last step
    return steps


def textdata_table(fpath, delim=" "):
    """Return a pandas DataFrame from a text data file.

    The returned table has 3 columns for metadata: "Step", "Time", and
    "Item".  "Step" refers to time step index, with 1 being the first
    time step taken by the solver (FEBio doesn't write the reference
    state to its text data files).  "Time" is the solution time at the
    time step.  "Item" is the item IDs of the nodes, elements, rigid
    bodies, or rigid connectors whose data is recorded in the text data
    file.

    There is also one column for each variable in the text data file.

    The function corrects FEBio's quirk of restarting the time step
    count at the start of each analysis step (corresponding to the
    <Step> element in FEBio XML), such that the returned step values
    increase sequentially with time.

    """
    with open(fpath, 'r') as f:
        # Find column names
        colnames = None
        for l in f:
            if l.startswith("*Data  = "):
                colnames = l.lstrip("*Data  = ").rstrip().split(";")
            elif not l.startswith("*"):
                break
        if colnames is None:
            raise ValueError(f"Could not find '*Data' row in {f.name}")
        f.seek(0)
        # Read the data
        rows = []
        for l in f:
            if l.startswith("*"):
                if l.startswith("*Step"):
                    step = l.lstrip("*Step  = ").rstrip()
                elif l.startswith("*Time"):
                    t = l.lstrip("*Time  = ").rstrip()
            else:  # numeric data line
                row = [step, t] + l.split(delim)
                # row[0] = int(row[0])
                rows.append(row)
    tab = pd.DataFrame(rows, dtype=float)
    tab.columns = ["Step", "Time", "Item"] + colnames
    tab["Step"] = tab["Step"].astype(int)
    tab["Item"] = tab["Item"].astype(int)
    # Force step IDs to increase sequentially with elapsed time
    tab = tab.sort_values("Time")
    idxs = np.where(tab["Step"].diff() < 0)[0]
    values = tab["Step"].values
    for idx in idxs:
        values[idx:] += values[idx-1]
    tab["Step"] = values
    return tab


class FebReader:
    """Read an FEBio xml file.

    """
    def __init__(self, file):
        """Read a file path as an FEBio xml file.

        """
        self.file = file
        self.root = normalize_xml(ET.parse(self.file).getroot())
        # Remove comments so iteration over child elements doesn't get
        # tripped up
        ET.strip_tags(self.root, ET.Comment)
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
            material = mat_obj_from_elemd(material)

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
                    extend = 'constant'  # default
                # Set interpolation
                if 'type' in e_lc.attrib:
                    interp = e_lc.attrib['type']
                else:
                    interp = 'linear'  # default
                # Create and store the Sequence object
                self._sequences[ord_id] = Sequence(curve, interp=interp, extend=extend)
        return self._sequences


    def _read_material(self, tag):
        """Get material properties dictionary from <material>.

        tag := the XML <material> element.

        """
        # TODO: This is a bad way of reading materials.  FEBio XML
        # doesn't have a lot of regularity and this conversion of a
        # material's XML tree to a dictionary is an extra step that just
        # gets in the way.
        m = {}
        m['material'] = tag.attrib['type']
        m['properties'] = {}
        constituents = []
        for child in tag:
            if child.tag in ['material', 'solid']:
                # Child element is a material
                constituents.append(self._read_material(child))
            if child.tag == "generation":
                t0 = _to_number(child.find("start_time").text)
                m["properties"].setdefault("start times", []).append(t0)
                e_mat = child.find("solid")
                constituents.append(self._read_material(e_mat))
            else:
                # Child element is a property (note: this isn't always true)
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

    def _read_rigid_body_element(self, model, e_rigid_body, explicit_bodies,
                                 implicit_bodies, step_id):
        """Read & apply a <rigid_body> element."""
        # Each <rigid_body> element defines constraints for one rigid
        # body, identified by its material ID.  Constraints may be fixed
        # (atemporal) or time-varying (temporal).
        #
        # Get the Body object from the material id
        mat_id = int(e_rigid_body.attrib["mat"]) - 1
        if mat_id in explicit_bodies:
            body = explicit_bodies[mat_id]
        else:
            # Assume mat_id refers to an implicit rigid body
            body = implicit_bodies[mat_id]
        # Read the body's constraints
        for e_dof in e_rigid_body.findall("fixed"):
            dof = DOF_NAME_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
            var = VAR_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
            model.fixed["body"][(dof, var)].add(body)
        for e_dof in e_rigid_body.findall("prescribed"):
            dof = DOF_NAME_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
            var = VAR_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
            seq = _read_parameter(e_dof, self.sequences)
            if e_dof.get("type", None) == "relative":
                relative = True
            else:
                relative = False
            model.apply_body_bc(body, dof, var, seq, step_id=step_id,
                                relative=relative)

    def model(self):
        """Return model.

        """
        # Get the materials dictionary so we can assign materials to
        # elements when we read the geometry
        materials_by_id, material_labels = self.materials()
        # Create model geoemtry
        mesh = self.mesh((materials_by_id, material_labels))
        model = Model(mesh)

        # Read Environment Constants
        model.environment = {}
        e_temperature = _find_unique_tag(self.root, "Globals/Constants/T")
        if e_temperature is not None:
            model.environment["temperature"] = _to_number(e_temperature.text)

        # Read Universal Constants.  These will eventually be superseded
        # by units support.
        model.constants = {}
        e_R = _find_unique_tag(self.root, "Globals/Constants/R")
        if e_R is not None:
            model.constants["R"] = _to_number(e_R.text)
        e_Fc = _find_unique_tag(self.root, "Globals/Constants/Fc")
        if e_Fc is not None:
            model.constants["F"] = _to_number(e_Fc.text)

        # Store the materials and their labels, now that the Model
        # object has been instantiated
        for ord_id, material in materials_by_id.items():
            name = material_labels[ord_id]
            model.named["materials"].add(name, material)
            model.named["materials"].add(ord_id, material, nametype="ordinal_id")
        materials_used = set(e.material for e in model.mesh.elements)
        # Read and store named sets of geometry
        named_sets = febioxml.read_named_sets(self.root)
        for entity_type in named_sets:
            for name in named_sets[entity_type]:
                obj = named_sets[entity_type][name]
                model.named[entity_type].add(name, obj)

        # From <Materials>, read heterogeneous local basis encoded using
        # local node IDs.
        e_mcs_local = self.root.findall('Material//mat_axis[@type="local"]')
        if e_mcs_local:
            # Check if there are multiple <mat_axis type="local">
            # elements; we can't support more than one unless they are
            # all equal.
            v = _vec_from_text(e_mcs_local[0].text)  # tuple
            equal = (_vec_from_text(e.text) == v
                     for e in e_mcs_local)
            if not all(equal):
                msg = f'{e_mat.base}:{e_mat.sourceline} Multiple <mat_axis type="local"> elements with unequal values are present.  febtools does not support this case.'
                raise ValueError(msg)
            # Convert the node ID encoded local basis to an explicit
            # basis for each finite element.  We can use the first
            # <mat_axis> element because we have just confirmed they are
            # all the same.
            elements_by_mat = {}
            for element in model.mesh.elements:
                elements_by_mat.setdefault(element.material, []).append(element)
            for e_mcs in e_mcs_local:
                mat_id = int(e_mcs.xpath("ancestor::material")[0].attrib["id"]) - 1
                mat = materials_by_id[mat_id]
                ids = _vec_from_text(e_mcs_local[0].text)  # 1-indexed
                for e in elements_by_mat[mat]:
                    e.basis = febioxml.basis_mat_axis_local(e, ids)

        # Read explicit rigid bodies.  Create a Body object for each
        # rigid body "material" in the XML with explicit geometry.
        explicit_bodies = {}
        for e_elements in self.root.findall("Geometry/Elements"):
            mat_id = int(e_elements.attrib["mat"]) - 1
            mat = model.named["materials"].obj(mat_id, nametype="ordinal_id")
            if isinstance(mat, material_lib.RigidBody):
                elements = []
                for e_elem in e_elements:
                    eid = int(e_elem.attrib["id"]) - 1
                    elements.append(model.mesh.elements[eid])
                explicit_bodies[mat_id] = Body(elements)

        # Read (1) implicit rigid bodies and (2) rigid body ↔ node set
        # rigid interfaces.
        implicit_bodies = {}
        for e_impbod in self.root.findall("Boundary/rigid"):
            # <rigid> elements may refer to implicit rigid bodies or to
            # rigid interfaces.  If the rigid "material" referenced by
            # the <rigid> element is assigned to elements, the element
            # represents a rigid interface.  Otherwise it represents an
            # rigid interface that interfaces with itself; i.e., an
            # implicit rigid body.
            mat_id = int(e_impbod.attrib["rb"]) - 1
            mat = model.named["materials"].obj(mat_id, nametype="ordinal_id")
            node_set = model.named["node sets"].obj(e_impbod.attrib["node_set"])
            if mat in materials_used:
                # This <rigid> element represents an explicit rigid body
                # ↔ node set interface.
                rigid_interface = RigidInterface(mat, node_set)
                model.constraints.append(rigid_interface)
            else:
                # This <rigid> element represents an implicit rigid body
                implicit_bodies[mat_id] = ImplicitBody(model.mesh, node_set, mat)

        # Read fixed boundary conditions. TODO: Support solutes
        #
        # Here, no prefix on an axis / BC name means it's named as in
        # febtools.  An `xml` prefix means it's named as in FEBio XML.
        #
        # Read fixed constraints on node sets:
        for e_fix in self.root.findall("Boundary/fix"):
            # Each <fix> tag may specify multiple bc labels.  Split them
            # up and convert each to febtools naming convention.
            if self.feb_version == '2.0':
                # In FEBio XML 2.0, bc labels are concatenated.
                fixed = febioxml_2_0.split_bc_names(e_fix.attrib['bc'])
            elif self.feb_version == '2.5':
                # In FEBio XML 2.5, bc labels are comma-delimeted.
                fixed = febioxml_2_5.split_bc_names(e_fix.attrib['bc'])
            # For each DoF, apply the fixed BCs to the model.
            for xml_bc in fixed:
                dof = DOF_NAME_FROM_XML_NODE_BC[xml_bc]
                # ^ `dof` is 'x1', 'x2', 'x3', 'α1', 'fluid', etc.
                var = VAR_FROM_XML_NODE_BC[xml_bc]
                # ^ `var` is 'displacement', 'rotation', 'pressure', etc.
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
                if not model.fixed['node'][(dof, var)]:
                    # If there is no node set assigned to this dof yet,
                    # simply re-use the node set.  This will preserve
                    # the node set's name if the model is re-exported.
                    model.fixed['node'][(dof, var)] = node_ids
                else:
                    # We are changing the node set, so existing
                    # references to it may become semantically invalid.
                    # And we can't remove the node set from the name
                    # registry because then later elements won't be
                    # interpretable.  So we must create a new node set.
                    model.fixed['node'][(dof, var)] =\
                        NodeSet(model.fixed['node'][(dof, var)] | node_ids)

        # Read global constraints on rigid bodies
        for e_rb in self.root.findall("Boundary/rigid_body"):
            self._read_rigid_body_element(model, e_rb, explicit_bodies,
                                          implicit_bodies, step_id=None)

        # Load curves (sequences)
        for seq_id, seq in self.sequences.items():
            model.named["sequences"].add(seq_id, seq, nametype="ordinal_id")

        # Steps
        model.steps = []
        # Find default module defined outside <Step>
        e_module = self.root.find("Module")
        if e_module is not None:
            top_module = e_module.attrib["type"]
        else:
            top_module = None
        # Read the <Step> elements
        for step_id, e_step in enumerate(self.root.findall('Step')):
            step_name = e_step.attrib["name"] if "name" in e_step.attrib else None
            # Module
            e_module = e_step.find('Module')
            if e_module is not None:
                module = e_module.text
            elif top_module is not None:
                module = top_module
            else:
                module = "solid"  # TODO: Pick default based on
                                  # materials
            # Control section
            control = {}
            e_control = e_step.find('Control')
            for e in e_control:
                nm = control_tagnames_from_febio[e.tag]
                if nm == "analysis type":
                    val = e.attrib['type']
                elif nm in control_values_from_febio:
                    val = control_values_from_febio[nm][e.text]
                else:
                    val = _maybe_to_number(e.text)
                control[nm] = val
            # Control/time_stepper section
            control['time stepper'] = {}
            e_stepper = e_control.find('time_stepper')
            for e in e_stepper:
                if e.tag in control_tagnames_from_febio:
                    k = control_tagnames_from_febio[e.tag]
                    control['time stepper'][k] = _read_parameter(e, self.sequences)
            # Add the step.  Use the method to get correct defaults;
            # e.g, correct keys for missing boundary conditions.
            model.add_step(module=module, control=control, name=step_name)
            #
            # Read body loadings
            for e_rb in e_step.findall("Boundary/rigid_body"):
                self._read_rigid_body_element(model, e_rb, explicit_bodies,
                                              implicit_bodies, step_id=step_id)
            # Node loadings are handled later

        # Read prescribed nodal conditions.
        #
        # Has to go after steps are created; otherwise there are no
        # steps to which to attach the applied conditions.
        if self.feb_version == "2.5":
            for condition in febioxml_2_5.iter_node_conditions(self.root):
                nodes = model.named["node sets"].obj(condition["node set name"])
                seq = model.named["sequences"].obj(condition["sequence ID"],
                                                   nametype="ordinal_id")
                # Check if we need a scaled sequence
                if condition["scale"] != 1.0:
                    seq = ScaledSequence(seq, condition["scale"])
                model.apply_nodal_bc(nodes, condition["dof"],
                                     condition["variable"],
                                     seq,
                                     scales=condition["nodal values"],
                                     relative=condition["relative"],
                                     step_id=condition["step ID"])

        # Output variables
        output_variables = []
        for e_var in self.root.findall("Output/plotfile/var"):
            output_variables.append(e_var.attrib["type"])
        model.output["variables"] = output_variables

        return model

    def mesh(self, material_info=None):
        """Return mesh.

        """
        if material_info is None:
            materials, mat_labels = self.materials()
        else:
            materials, mat_labels = material_info
        nodes = np.array([[float(a) for a in b.text.split(",")]
                          for b in self.root.findall("./Geometry/Nodes/*")])
        # Read elements
        elements = []  # nodal index format
        for elset in self.root.findall("./Geometry/Elements"):
            # TODO: Allow loading meshes that have no material.
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
        #
        # Read <MeshData> (partial support; just for ElementData with
        # var="mat_axis")
        for e_edata in self.root.findall("MeshData/ElementData[@var='mat_axis']"):
            # Get the name of the referenced element set
            try:
                nm_eset = e_edata.attrib["elem_set"]
            except KeyError:
                raise ValueError(f"{e_edata.base}:{e_edata.sourceline} <ElementData> is missing its required 'elem_set' attribute.")
            # Get the referenced element set
            e_eset = self.root.find(f"Geometry/ElementSet[@name='{nm_eset}']")
            if e_eset is None:
                raise ValueError(f"{e_edata.base}:{e_edata.sourceline} <ElementData> references an element set named '{nm_eset}', which is not defined.")
            eset_elements = tuple(e_eset.iterchildren())
            for e_elem in e_edata:
                a = _vec_from_text(e_elem.find("a").text)
                d = _vec_from_text(e_elem.find("d").text)
                basis = orthonormal_basis(a, d)
                leid = int(e_elem.attrib["lid"]) - 1
                eid = int(eset_elements[leid].attrib["id"]) - 1
                # Who thought this much indirection in a data file
                # format was a good idea?
                mesh.elements[eid].basis = basis
        return mesh


def mat_obj_from_elemd(d):
    """Convert material element to material object"""
    # Set default values for common properties
    orientation = None
    # TODO: Handle conflicting orientations; e.g., both <mat_axis> and
    # <fiber>.  FEBio stacks these.
    # Do we even support reading this material?
    if not d["material"] in febioxml.solid_class_from_name:
        raise ValueError(f"{d['material']} is not supported in the loading of FEBio XML.")

    # Read material orientation
    #
    # Read material orientation in the form of <mat_axis> or <fiber>
    p_mat_axis = d["properties"].pop("mat_axis", None)
    p_fiber = d["properties"].pop("fiber", None)
    if p_mat_axis is not None and p_fiber is not None:
        # FEBio's documentation says that only one could be defined, but
        # FEBio itself accepts both, with undocumented handling (e.g.,
        # precedence).  So raise an error.
        raise ValueError(f"Found both <mat_axis> and <fiber> XML elements in {d['material']}; only one may be present.")
    if p_mat_axis is not None:
        if p_mat_axis["type"] == "vector":
            orientation = orthonormal_basis(p_mat_axis["a"], p_mat_axis["d"])
        # <mat_axis type="local"> is converted to a heterogeneous
        # orientation in Model's initializer; no need to handle it here.
    elif p_fiber is not None:
        # Currently we only support <fiber type="angles">.  Other
        # types: local, vector, spherical, cylindrical.
        if p_fiber["type"] == "angles":
            orientation = vec_from_sph(p_fiber["theta"], p_fiber["phi"])
        elif p_fiber["type"] == "vector":
            orientation = np.array(p_fiber["value"])
        else:
            raise NotImplementedError
    # Read material orientation in the form of material property-like
    # XML elements.
    p_theta = d["properties"].pop("theta", None)
    p_phi = d["properties"].pop("phi", None)
    # Verify that both spherical angles are present, or neither
    if p_theta is not None and p_phi is None:
        raise ValueError(f"Found a <theta> element but no <phi> in {d['material']}; both spherical angles are required to define a material orientation.")
    if p_theta is None and p_phi is not None:
        raise ValueError(f"Found a <phi> element but no <theta> in {d['material']}; both spherical angles are required to define a material orientation.")
    if p_theta is not None and p_phi is not None:
        matprop_orientation = vec_from_sph(p_theta, p_phi)
        if orientation is None:
            orientation = matprop_orientation
        else:
            # Have to combine orientations
            if np.array(orientation).ndim == 2:
                orientation = orientation @ matprop_orientation
            else:
                # `orientation` is just a vector.  Interpret it as
                # indicating a transformation from [1, 0, 0] to its
                # value.
                raise NotImplementedError

    ## Create material object
    cls = febioxml.solid_class_from_name[d["material"]]
    if d["material"] == "solid mixture":
        constituents = []
        for d_child in d["constituents"]:
            child_material = mat_obj_from_elemd(d_child)
            constituents.append(child_material)
        material = cls(constituents)
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
        solid_fraction = d["properties"]["phi0"]  # what is the default?
        # Return the Poroelastic Solid object
        material = material_lib.PoroelasticSolid(solid, permeability,
                                                 solid_fraction)
    elif d["material"] == "multigeneration":
        # Constructing materials for the list of generations works
        # just like a solid mixture
        constituents = []
        for d_child in d["constituents"]:
            constituents.append(mat_obj_from_elemd(d_child))
        generations = ((t, mat) for t, mat in
                       zip(d["properties"]["start times"], constituents))
        material = cls(generations)
    elif hasattr(cls, "from_feb") and callable(cls.from_feb):
        if "density" in d["properties"]:
            del d["properties"]["density"]  # density not supported yet
        material = cls.from_feb(**d["properties"])
    else:
        material = cls(d["properties"])
    # Apply total (sub)material orientation
    if orientation is not None:
        material = material_lib.OrientedMaterial(material, orientation)
    return material


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


    def __del__(self):
        self.f.close()


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
        var['domain variables'] = self._rdict('domain')
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
        data['element variables'] = data['domain variables']
        # ^ for backwards compatibility
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
                    fmt.append(xplt.value_layout_from_id[
                        struct.unpack(self.endian + 'I', data)[0]])
                elif label == 'item name':
                    name.append(data[:data.find(b'\x00')].decode())
                elif label == 'item array size':
                    # Do nothing.  This tag was added around FEBio 2.7
                    # to FEBio's `DICTIONARY_ITEM` class.  It's not
                    # clear what it does, but it's apparently 0 for
                    # everything but arrays.  (So why is it defined at
                    # all for not-arrays?)
                    pass
                else:
                    msg = f"`{label}` block not expected as a child of `dict_item`."
                    raise Exception(msg)
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

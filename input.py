import dataclasses
import os
from pathlib import Path
from typing import Any, Union, Dict, Tuple
import warnings

from lxml import etree
import struct
import numpy as np
from numpy import array
import pandas as pd

import waffleiron.element
from waffleiron.exceptions import UnsupportedFormatError
from operator import itemgetter

from .math import orthonormal_basis, vec_from_sph
from .model import Model, Mesh
from .core import (
    _canonical_face,
    ZeroIdxID,
    OneIdxID,
    Body,
    ContactConstraint,
    ImplicitBody,
    Interpolant,
    Extrapolant,
    Sequence,
    ScaledSequence,
    NodeSet,
    FaceSet,
    ElementSet,
    RigidInterface,
)
from .control import (
    Physics,
    auto_physics,
    Ticker,
    IterController,
    SaveIters,
    Solver,
    Step,
)
from .element import Element
from . import xplt
from . import febioxml, febioxml_2_0, febioxml_2_5, febioxml_3_0
from . import material as material_lib
from .febioxml import (
    VAR_FROM_XML_NODE_BC,
    DOF_NAME_FROM_XML_NODE_BC,
    SUPPORTED_FEBIO_XML_VERS,
    elem_cls_from_feb,
    normalize_xml,
    to_number,
    maybe_to_number,
    find_unique_tag,
    read_parameter,
    read_parameters,
    OptParameter,
    ReqParameter,
    text_to_bool,
)


def _nstrip(string):
    """Remove trailing nulls from string."""
    for i, c in enumerate(string):
        if c == "\x00":
            return string[:i]
    return string


def _vec_from_text(s) -> tuple:
    return tuple(to_number(x.strip()) for x in s.split(","))


def read_contacts(root, named_face_sets, febioxml_module):
    fx = febioxml_module
    global_contacts = []
    step_contacts = []
    for e in root.findall("Contact/contact"):
        global_contacts.append(read_contact(e, named_face_sets, febioxml_module))
    for e_Step in root.findall(f"{fx.STEP_PARENT}/{fx.STEP_NAME}"):
        contacts = []
        for e in e_Step.findall("Contact/contact"):
            contacts.append(read_contact(e, named_face_sets, febioxml_module))
        step_contacts.append(contacts)
    return global_contacts, step_contacts


def read_contact(e_contact: etree.Element, named_face_sets, febioxml_module):
    fx = febioxml_module
    tree = e_contact.getroottree()
    root = tree.getroot()
    surf_pair = e_contact.attrib["surface_pair"]
    e_SurfacePair = find_unique_tag(
        root, f"{fx.MESH_PARENT}/SurfacePair[@name='{surf_pair}']"
    )
    e_leader = find_unique_tag(e_SurfacePair, fx.SURFACEPAIR_LEADER_NAME)
    e_follower = find_unique_tag(e_SurfacePair, fx.SURFACEPAIR_FOLLOWER_NAME)
    leader = named_face_sets.obj(fx.get_surface_name(e_leader))
    follower = named_face_sets.obj(fx.get_surface_name(e_follower))
    # Read simple parameters
    kwargs = read_parameters(e_contact, fx.CONTACT_PARAMS)
    # "two_pass" requires special handling
    if (e_two_pass := e_contact.find("two_pass")) is not None:
        two_pass = text_to_bool(e_two_pass.text)
        if two_pass:
            kwargs["passes"] = 2
        else:
            kwargs["passes"] = 1
    contact = ContactConstraint(
        leader, follower, algorithm=e_contact.attrib["type"], **kwargs
    )
    return contact


def read_febio_xml(f):
    """Return lxml tree for FEBio XML file path or IO object."""
    parser = etree.XMLParser(remove_blank_text=True)
    try:
        f = open(f, "rb")
    except TypeError:
        # Assume pth is already an IO object; caller may have some reason to
        # hold it open, or just never wrote the file to disk.
        tree = etree.parse(f, parser)
    else:
        try:
            tree = etree.parse(f, parser)
        finally:
            f.close()
    return tree


def read_mesh(root: etree.Element, febioxml_module) -> Tuple[array, array]:
    """Return lists of nodes and elements

    Materials will *not* be assigned.

    """
    fx = febioxml_module
    # Read nodes
    nodes = np.array(
        [
            [float(a) for a in b.text.split(",")]
            for b in root.findall(f"./{fx.MESH_PARENT}/Nodes/*")
        ]
    )
    # Read elements
    elements = []  # nodal index format
    for elset in root.findall(f"./{fx.MESH_PARENT}/Elements"):
        # map element type strings to classes
        cls = elem_cls_from_feb[elset.attrib["type"]]
        for elem in elset.findall("./elem"):
            ids = [ZeroIdxID(int(a) - 1) for a in elem.text.split(",")]
            e = cls.from_ids(ids, nodes)  # type: ignore
            elements.append(e)
    return nodes, elements


def read_named_sets(root: etree.Element, febioxml_module) -> Dict[str, Dict[str, list]]:
    """Read nodesets, etc., and apply them to a model."""
    fx = febioxml_module
    sets: Dict[str, dict] = {"node sets": {}, "face sets": {}, "element sets": {}}
    tag_name = {
        "node sets": "NodeSet",
        "face sets": "Surface",
        "element sets": "ElementSet",
    }
    # Handle items that are stored by id
    for k in ["node sets", "element sets"]:
        for e_set in root.findall(f"./{fx.MESH_PARENT}/{tag_name[k]}"):
            items = [
                ZeroIdxID(int(e_item.attrib["id"]) - 1)
                for e_item in e_set.getchildren()
            ]
            sets[k][e_set.attrib["name"]] = items
    # Handle items that are stored as themselves
    for k in ["face sets"]:
        for tag_set in root.findall(f"./{fx.MESH_PARENT}/{tag_name[k]}"):
            items = [
                _canonical_face(
                    [ZeroIdxID(int(s.strip()) - 1) for s in tag_item.text.split(",")]
                )
                for tag_item in tag_set.getchildren()
            ]
            sets[k][tag_set.attrib["name"]] = items
    return sets


def read_step(step_xml, model, physics, febioxml_module):
    """Return a Step object for a <Step> XML element

    If an optional control parameter are missing, it will be initialized
    with the default value as documented in the FEBio user manual.  This
    may differ from the default actually used by FEBio.

    This function does not (yet) read conditions, only the control
    settings (time and solver).

    """
    # The model object is required to resolve named entities (including,
    # at least, sequences and rigid bodies) that are referenced by the
    # simulation step.

    fx = febioxml_module

    step_name = step_xml.attrib["name"] if "name" in step_xml.attrib else None

    ticker_kwargs = read_parameters(step_xml, fx.TICKER_PARAMS)
    solver_kwargs = read_parameters(step_xml, fx.SOLVER_PARAMS)
    controller_kwargs = read_parameters(step_xml, fx.CONTROLLER_PARAMS)
    # Must points, and hence dtmax, take special handling
    e = find_unique_tag(step_xml, "Control/time_stepper/dtmax")
    if e is not None:
        ticker_kwargs["dtmax"] = read_parameter(e, model.named["sequences"])
    else:
        ticker_kwargs["dtmax"] = fx.TICKER_PARAMS["dtmax"].default
        controller_kwargs["save_iters"] = SaveIters.MAJOR  # FEBio default
    ticker = Ticker(**ticker_kwargs)
    controller = IterController(**controller_kwargs)
    solver = Solver(**solver_kwargs)
    # update_method requires custom conversion
    update_method = {"0": "BFGS", "1": "Broyden", "BFGS": "BFGS", "BROYDEN": "Broyden"}
    if not solver.update_method in ("BFGS", "Broyden"):
        # ^ could have gotten a default value from Solver.__init__
        solver.update_method = update_method[solver.update_method]
    step = Step(physics=physics, ticker=ticker, solver=solver, controller=controller)

    return step, step_name


def load_model(fpath):
    """Loads a model (feb) and the solution (xplt) if it exists.



    The following data is supported for FEBio XML 2.0:
    - Materials
    - Geometry: nodes and elements

    """
    if isinstance(fpath, str):
        fpath = Path(fpath)
    # Don't try to load a model from a nonexistent file
    if not fpath.exists():
        raise ValueError(
            f"{fpath} does not appear to exist.  The working directory was {os.getcwd()}."
        )
    fp_feb = fpath.with_suffix(".feb")
    fp_xplt = fpath.with_suffix(".xplt")
    # Attempt to read the FEBio xml file
    try:
        model = FebReader(str(fp_feb)).model()
        feb_ok = True
    except UnsupportedFormatError as err:
        # The .feb file is some unsupported version
        msg = (
            f"{err.message}.  Falling back to defining the model from the .xplt "
            "file alone.  Values given only in the .feb file will not be "
            "available."
        )
        warnings.warn(msg)
        feb_ok = False
    # Attempt to read the xplt file, if it exists
    if os.path.exists(fp_xplt):
        with open(fp_xplt, "rb") as f:
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
        raise ValueError(
            "Neither `{}` nor `{}` could be read.  Check that they exist "
            "and are accessible.".format(fp_feb, fp_xplt)
        )
    model.name = fpath.stem
    return model


def textdata_list(fpath, delim=" "):
    """Reads FEBio logfile as a list of the steps' data.

    The function returns a list of dictionaries, one per solution step.
    The keys are the variable names (e.g. x, sxy, J, etc.). Each
    dictionary value is a list of variable values over all the nodes.
    The indexing of this list is the same as in the original file.

    This function can be used for both node and element data.

    """
    with open(fpath, "r") as f:
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
            if l.startswith("*"):
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
                        v = float("nan")
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
    with open(fpath, "r") as f:
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
        values[idx:] += values[idx - 1]
    tab["Step"] = values
    return tab


class FebReader:
    """Read an FEBio xml file."""

    def __init__(self, file):
        """Read a file path as an FEBio xml file."""
        self.file = file
        self.root = normalize_xml(etree.parse(self.file).getroot())
        # Remove comments so iteration over child elements doesn't get
        # tripped up
        etree.strip_tags(self.root, etree.Comment)
        self.feb_version = self.root.attrib["version"]
        if self.root.tag != "febio_spec":
            raise Exception(
                "Root node is not 'febio_spec': '"
                + file.name
                + "is not an FEBio xml file."
            )
        if self.feb_version not in SUPPORTED_FEBIO_XML_VERS:
            msg = f"FEBio XML version {self.feb_version} is not supported by waffleiron"
            raise UnsupportedFormatError(msg, file, self.feb_version)
        # Get the correct FEBio XML module
        version_major, version_minor = [int(a) for a in self.feb_version.split(".")]
        if version_major == 2 and version_minor == 0:
            self.febioxml_module = febioxml_2_0
        elif version_major == 2 and version_minor == 5:
            self.febioxml_module = febioxml_2_5
        elif version_major == 3 and version_minor == 0:
            self.febioxml_module = febioxml_3_0
        else:
            raise NotImplementedError(
                f"Writing FEBio XML {version_major}.{version_minor} is not supported."
            )
        self._sequences = None  # memo for sequences()

    def materials(self):
        """Return dictionary of materials keyed by id."""
        mats = {}
        mat_labels = {}
        for m in self.root.findall("./Material/material"):
            # Read material into dictionary
            material = self._read_material(m)
            mat_id = int(m.attrib["id"]) - 1  # FEBio counts from 1
            material = mat_obj_from_elemd(material)

            # Store material in index
            mats[mat_id] = material
            if "name" in m.attrib:
                mat_labels[mat_id] = m.attrib["name"]
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
        fx = self.febioxml_module
        if self._sequences is None:
            self._sequences = fx.sequences(self.root)
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
        m["material"] = tag.attrib["type"]
        m["properties"] = {}
        constituents = []
        for child in tag:
            if child.tag in ["material", "solid"]:
                # Child element is a material
                constituents.append(self._read_material(child))
            if child.tag == "generation":
                t0 = to_number(child.find("start_time").text)
                m["properties"].setdefault("start times", []).append(t0)
                e_mat = child.find("solid")
                constituents.append(self._read_material(e_mat))
            else:
                # Child element is a property (note: this isn't always true)
                m["properties"][child.tag] = self._read_property(child)
        if constituents:
            m["constituents"] = constituents
        return m

    def _read_property(self, tag):
        """Read a material property element."""
        # Check if this is a time-varying or fixed property
        if "lc" in tag.attrib:
            # The property is time-varying
            seq_id = int(tag.attrib["lc"]) - 1
            sequence = self.sequences[seq_id]
            scale = to_number(tag.text)
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
                    v = [float(a) for a in v.split(",")]
                    if len(v) == 1:
                        v = v[0]
                    if not p:
                        return v
                    else:
                        p["value"] = v
            return p

    def model(self):
        """Return model."""
        fx = self.febioxml_module

        # Read stuff that doesn't depend on other stuff.

        # Create model geometry.
        mesh = self.mesh()
        model = Model(mesh)
        domains = fx.read_domains(self.root)

        # Read Universal Constants.  These will eventually be superseded
        # by units support.
        model.constants = {}
        e_R = find_unique_tag(self.root, "Globals/Constants/R")
        if e_R is not None:
            model.constants["R"] = to_number(e_R.text)
        e_Fc = find_unique_tag(self.root, "Globals/Constants/Fc")
        if e_Fc is not None:
            model.constants["F"] = to_number(e_Fc.text)

        # Load curves (sequences)
        for seq_id, seq in self.sequences.items():
            model.named["sequences"].add(seq_id, seq, nametype="ordinal_id")

        # Read named sets of geometric entities (node sets, element
        # sets, face sets) as *ordered lists*.  FEBio XML sometimes
        # makes references into an entity sets using the positional
        # index of its constituent entitites, so order must be preserved
        # while we are reading the XML file.
        named_sets = read_named_sets(self.root, self.febioxml_module)
        cls_from_entity_type = {
            "node sets": NodeSet,
            "face sets": FaceSet,
            "element sets": ElementSet,
        }
        for entity_type in named_sets:
            cls = cls_from_entity_type[entity_type]
            for name in named_sets[entity_type]:
                ordered = named_sets[entity_type][name]
                unordered = cls(ordered)
                model.named[entity_type].add(name, unordered)

        # Read stuff that might depend on other stuff.

        # Materials.  May include time-varying parameters and hence may
        # depend on sequences.
        materials_by_id, material_labels = self.materials()
        for ord_id, material in materials_by_id.items():
            name = material_labels[ord_id]
            model.named["materials"].add(name, material)
            model.named["materials"].add(ord_id, material, nametype="ordinal_id")
        # Apply materials to elements, keeping track of which materials
        # were used this way
        materials_used = set()
        for domain in domains:
            nametype, name = domain["material"]
            material = model.named["materials"].obj(name, nametype)
            materials_used.add(material)
            for i in domain["elements"]:
                model.mesh.elements[i].material = material

        # Element data: material axes
        for e_edata in self.root.findall(
            f"{fx.ELEMENTDATA_PARENT}/ElementData[@var='mat_axis']"
        ):
            try:
                nm_eset = e_edata.attrib["elem_set"]
            except KeyError:
                raise ValueError(
                    f"{e_edata.base}:{e_edata.sourceline} <ElementData> is missing its required 'elem_set' attribute."
                )
            elementsets = named_sets["element sets"]
            try:
                elementset = elementsets[nm_eset]
            except KeyError:
                raise ValueError(
                    f"{e_edata.base}:{e_edata.sourceline} <ElementData> references an element set named '{nm_eset}', which is not defined."
                )
            for e in e_edata.findall("elem"):
                a = _vec_from_text(e.find("a").text)
                d = _vec_from_text(e.find("d").text)
                basis = orthonormal_basis(a, d)
                idx = ZeroIdxID(int(e.attrib["lid"]) - 1)
                # ^ positional index in element set list
                id_ = elementset[idx]
                # Who thought this much indirection in a data file
                # format was a good idea?
                mesh.elements[id_].basis = basis

        # Find out what physics are used by the model.  This is a
        # required attribute in FEBio XML 3.0, both as documented and in
        # practice.  FEBio 2 ignored a missing <Module> element though,
        # so in practice it may be missing.  In this case we deduce the
        # appropriate physics instead of failing, which requires the
        # materials list (and possibly, in the future, other
        # information).
        e_module = self.root.find("Module")
        if e_module is not None:
            physics = Physics(e_module.attrib["type"])
        else:
            # Deduce the implied physics
            physics = auto_physics([m for m in materials_by_id.values()])

        # Read Environment Constants.  These could, in principle be
        # time-varying (depend on a sequence), despite the name
        # "constant".
        model.environment = {}
        e_temperature = find_unique_tag(self.root, "Globals/Constants/T")
        if e_temperature is not None:
            model.environment["temperature"] = to_number(e_temperature.text)

        # From <Materials>, read heterogeneous local basis encoded using
        # local node IDs.
        e_mcs_local = self.root.findall('Material//mat_axis[@type="local"]')
        if e_mcs_local:
            # Check if there are multiple <mat_axis type="local">
            # elements; we can't support more than one unless they are
            # all equal.
            v = _vec_from_text(e_mcs_local[0].text)  # tuple
            equal = (_vec_from_text(e.text) == v for e in e_mcs_local)
            if not all(equal):
                msg = f'{e_mcs_local.base}:{e_mcs_local.sourceline} Multiple <mat_axis type="local"> elements with unequal values are present.  waffleiron does not support this case.'
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
        for domain in domains:
            nametype, name = domain["material"]
            material = model.named["materials"].obj(name, nametype)
            ids = model.named["materials"].names(material, "ordinal_id")
            assert (len(ids)) == 1
            mat_id = ids[0]
            if isinstance(material, material_lib.RigidBody):
                elements = []
                for i in domain["elements"]:
                    elements.append(model.mesh.elements[i])
                explicit_bodies[mat_id] = Body(elements)

        # Read (1) implicit rigid bodies and (2) rigid body ↔ node set
        # rigid interfaces.
        implicit_bodies = {}
        for e_impbod in self.root.findall(f"{fx.IMPBODY_PARENT}/{fx.IMPBODY_NAME}"):
            # <rigid> elements may refer to implicit rigid bodies or to
            # rigid interfaces.  If the rigid "material" referenced by
            # the <rigid> element is assigned to elements, the element
            # represents a rigid interface.  Otherwise it represents an
            # rigid interface that interfaces with itself; i.e., an
            # implicit rigid body.
            nodeset_name, mat_id = fx.parse_rigid_interface(e_impbod)
            mat = model.named["materials"].obj(mat_id, nametype="ordinal_id")
            node_set = model.named["node sets"].obj(nodeset_name)
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
        # waffleiron.  An `xml` prefix means it's named as in FEBio XML.
        #
        # Read fixed constraints on node sets:
        fixed_node_bcs = fx.read_fixed_node_bcs(self.root, model)
        for (dof, var), nodeset in fixed_node_bcs.items():
            model.fixed["node"][(dof, var)] = nodeset
        #
        # Read global constraints on rigid bodies.  Needs to come after
        # reading sequences, or the relevant sequences won't be in the
        # sequence registry.
        for e_rb in self.root.findall(f"{fx.BODY_COND_PARENT}/{fx.BODY_COND_NAME}"):
            fx.apply_body_bc(model, e_rb, explicit_bodies, implicit_bodies, step=None)

        # Steps
        for e_step in self.root.findall(f"{fx.STEP_PARENT}/{fx.STEP_NAME}"):
            step, name = read_step(e_step, model, physics, self.febioxml_module)
            model.add_step(step, name)
        # Rigid body boundary conditions
        for e_step, (step, name) in zip(
            self.root.findall(f"{fx.STEP_PARENT}/{fx.STEP_NAME}"), model.steps
        ):
            for e_rb in e_step.findall(f"{fx.BODY_COND_PARENT}/{fx.BODY_COND_NAME}"):
                fx.apply_body_bc(model, e_rb, explicit_bodies, implicit_bodies, step)
        # Prescribed nodal conditions.  Has to go after steps are
        # created; otherwise there are no steps to which to attach the
        # applied conditions.
        for condition in fx.iter_node_conditions(self.root):
            if condition["node set name"] is not None:
                nodes = model.named["node sets"].obj(condition["node set name"])
            else:
                # In some FEBio XML formats, nodal values can be set
                # directly in the boundary condtion, without an named
                # node set as intermediary.
                nodes = list(condition["nodal values"].keys())
            seq = model.named["sequences"].obj(
                condition["sequence ID"], nametype="ordinal_id"
            )
            # Check if we need a scaled sequence
            if condition["scale"] is not None:
                seq = ScaledSequence(seq, condition["scale"])
            model.apply_nodal_bc(
                nodes,
                condition["dof"],
                condition["variable"],
                seq,
                scales=condition["nodal values"],
                relative=condition["relative"],
                step=model.steps[condition["step ID"]].step,
            )

        # Read contacts into steps
        global_contacts, step_contacts = read_contacts(
            self.root, model.named["face sets"], fx
        )
        for contact in global_contacts:
            model.add_contact(contact)
        for i, step_list in enumerate(step_contacts):
            for contact in step_list:
                model.add_contact(contact, step_idx=i)

        # Output variables
        output_variables = []
        for e_var in self.root.findall("Output/plotfile/var"):
            output_variables.append(e_var.attrib["type"])
        model.output["variables"] = output_variables

        return model

    def mesh(self):
        """Return mesh."""
        fx = self.febioxml_module
        nodes, elements = read_mesh(self.root, fx)
        mesh = Mesh(nodes, elements)
        return mesh


def mat_obj_from_elemd(d):
    """Convert material element to material object"""
    # Set default values for common properties
    orientation = None
    # TODO: Handle conflicting orientations; e.g., both <mat_axis> and
    # <fiber>.  FEBio stacks these.
    # Do we even support reading this material?
    if not d["material"] in febioxml.solid_class_from_name:
        raise ValueError(
            f"{d['material']} is not supported in the loading of FEBio XML."
        )

    # Read material orientation
    #
    # Read material orientation in the form of <mat_axis> or <fiber>
    p_mat_axis = d["properties"].pop("mat_axis", None)
    p_fiber = d["properties"].pop("fiber", None)
    if p_mat_axis is not None and p_fiber is not None:
        # FEBio's documentation says that only one could be defined, but
        # FEBio itself accepts both, with undocumented handling (e.g.,
        # precedence).  So raise an error.
        raise ValueError(
            f"Found both <mat_axis> and <fiber> XML elements in {d['material']}; only one may be present."
        )
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
        raise ValueError(
            f"Found a <theta> element but no <phi> in {d['material']}; both spherical angles are required to define a material orientation."
        )
    if p_theta is None and p_phi is not None:
        raise ValueError(
            f"Found a <phi> element but no <theta> in {d['material']}; both spherical angles are required to define a material orientation."
        )
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

    # Create material object
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
            raise ValueError(
                """A porelastic solid was encountered with {len(d['constituents'])} solid constituents.  Poroelastic solids must have exactly one solid constituent."""
            )
        solid = mat_obj_from_elemd(d["constituents"][0])
        solid_fraction = d["properties"]["phi0"]  # what is the default?
        # Return the Poroelastic Solid object
        material = material_lib.PoroelasticSolid(solid, permeability, solid_fraction)
    elif d["material"] == "multigeneration":
        # Constructing materials for the list of generations works
        # just like a solid mixture
        constituents = []
        for d_child in d["constituents"]:
            constituents.append(mat_obj_from_elemd(d_child))
        generations = (
            (t, mat) for t, mat in zip(d["properties"]["start times"], constituents)
        )
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

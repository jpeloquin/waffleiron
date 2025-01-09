import dataclasses
import os
from pathlib import Path
from typing import Any, List, Union, Dict, Tuple
import warnings

from lxml import etree
import struct
import numpy as np
from numpy import array
import pandas as pd
from numpy.typing import NDArray
from waffleiron import febioxml_4_0

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
    Dynamics,
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
    CONTACT_CLASS_FROM_XML,
    VAR_FROM_XML_NODE_BC,
    DOF_NAME_FROM_XML_NODE_BC,
    SUPPORTED_FEBIO_XML_VERS,
    VerbatimXMLMaterial,
    elem_cls_from_feb,
    normalize_xml,
    to_number,
    maybe_to_number,
    find_unique_tag,
    read_material,
    read_parameter,
    read_parameters,
    OptParameter,
    ReqParameter,
    to_bool,
    BodyConstraint,
    vector_from_text,
    ids_from_text,
)


def _nstrip(string):
    """Remove trailing nulls from string."""
    for i, c in enumerate(string):
        if c == "\x00":
            return string[:i]
    return string


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
    algo = e_contact.attrib["type"]
    # Read simple parameters
    cls = CONTACT_CLASS_FROM_XML[algo]
    kwargs = {
        k: v
        for k, v in read_parameters(e_contact, fx.CONTACT_PARAMS).items()
        if hasattr(cls, k)
    }
    contact = cls(leader, follower, **kwargs)
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


def read_mesh(root: etree.Element, febioxml_module) -> Tuple[NDArray, List]:
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
            items = fx.read_nodeset(e_set)
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

    If an optional control parameter is missing, it will be initialized with the
    default value as documented in the FEBio user manual.  This may differ from the
    default actually used by FEBio.

    This function does not (yet) read conditions, only the control
    settings (time and solver).

    """
    # The model object is required to resolve named entities (including,
    # at least, sequences and rigid bodies) that are referenced by the
    # simulation step.

    fx = febioxml_module

    step_name = step_xml.attrib["name"] if "name" in step_xml.attrib else None

    # Dynamics
    e = find_unique_tag(step_xml, "Control/analysis")
    if e is not None:
        dynamics = fx.read_dynamics(e)
    else:
        dynamics = Dynamics.STATIC

    ticker_kwargs = read_parameters(step_xml, fx.TICKER_PARAMS)
    controller_kwargs = read_parameters(step_xml, fx.CONTROLLER_PARAMS)
    # Must points, and hence dtmax, take special handling
    e = find_unique_tag(step_xml, "Control/time_stepper/dtmax")
    if e is not None:
        ticker_kwargs["dtmax"] = read_parameter(
            e, model.named["sequences"]["ordinal_id"]
        )
    else:
        ticker_kwargs["dtmax"] = fx.TICKER_PARAMS["dtmax"].default
        controller_kwargs["save_iters"] = SaveIters.MAJOR  # FEBio default
    ticker = Ticker(**ticker_kwargs)
    controller = IterController(**controller_kwargs)
    solver = fx.read_solver(step_xml)
    # update_method requires custom conversion
    update_method = {"0": "BFGS", "1": "Broyden", "BFGS": "BFGS", "BROYDEN": "Broyden"}
    if not solver.update_method in ("BFGS", "Broyden"):
        # ^ could have gotten a default value from Solver.__init__
        solver.update_method = update_method[solver.update_method]
    step = Step(
        physics=physics,
        dynamics=dynamics,
        ticker=ticker,
        solver=solver,
        controller=controller,
    )

    return step, step_name


def load_model(fpath, read_xplt=True, fallback_to_xplt=False):
    """Loads a model (feb) and the solution (xplt) if it exists.

    :param read_xplt: If True (default), try to read the xplt file (if any) with the
    same basename as the provided path.  Set to False if the FEBio XML file and the xplt
    file do not describe the same model.

    The following data is supported for FEBio XML 2.0:
    - Materials
    - Geometry: nodes and elements

    """
    if isinstance(fpath, str):
        fpath = Path(fpath)
    # Don't try to load a model from a nonexistent file
    if fpath.exists():
        if fpath.suffix == ".xplt":
            fp_feb = fpath.with_suffix(".feb")
            fp_xplt = fpath
        else:
            fp_feb = fpath
            fp_xplt = fpath.with_suffix(".xplt")
    else:
        raise ValueError(
            f"The provided path `{fpath}`, which resolves to `{fpath.resolve()}`, does not appear to exist."
        )
    # Attempt to read the FEBio xml file
    try:
        model = FebReader(fp_feb).model()
        feb_ok = True
    except UnsupportedFormatError as err:
        # The .feb file is some unsupported version
        if fallback_to_xplt:
            warnings.warn(
                f"{err.message}.  Falling back to defining the model from the .xplt file alone.  Values given only in the .feb file will not be available."
            )
            feb_ok = False
        else:
            raise err
    # Attempt to read the xplt file, if it exists
    if read_xplt:
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


def apply_body_bc(model: Model, step: Step, body_bcs: List[BodyConstraint]):
    """Apply body constraints to model

    :param model: Model to which to add the body constraints.

    :param step: Step to which to add the body constraints.

    :param body_bcs: List of body constraints to apply to the model.

    """
    for bc in body_bcs:
        if bc.constant:
            model.fixed["body"][(bc.dof, bc.variable)].add(bc.body)
        else:
            model.apply_body_bc(
                bc.body,
                bc.dof,
                bc.variable,
                bc.sequence,
                relative=bc.relative,
                step=step,
            )


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
    """Read an FEBio XML file."""

    def __init__(self, file):
        """Read a file path as an FEBio xml file."""
        self.file = str(file)
        self.root = normalize_xml(etree.parse(self.file).getroot())
        # Remove comments so iteration over child elements doesn't get
        # tripped up
        etree.strip_tags(self.root, etree.Comment)
        self.feb_version = self.root.attrib["version"]
        if self.root.tag != "febio_spec":
            raise ValueError(
                f"Root node is not 'febio_spec': {file} is not an FEBio XML file."
            )
        if self.feb_version not in SUPPORTED_FEBIO_XML_VERS:
            msg = f"FEBio XML version {self.feb_version} is not supported by waffleiron"
            raise UnsupportedFormatError(msg, file, self.feb_version)
        # Get the correct FEBio XML module
        self.febioxml_module = {
            "2.0": febioxml_2_0,
            "2.5": febioxml_2_5,
            "3.0": febioxml_3_0,
            "4.0": febioxml_4_0,
        }[self.feb_version]
        self._sequences = None  # memo for sequences()

    def materials(self):
        """Return materials and their labels keyed by id."""
        mats = {}
        mat_labels = {}
        for e in self.root.findall("./Material/material"):
            mat_id = int(e.attrib["id"]) - 1  # FEBio counts from 1
            mat = read_material(e, self.sequences)
            mats[mat_id] = mat
            if "name" in e.attrib:
                mat_labels[mat_id] = e.attrib["name"]
            else:
                mat_labels[mat_id] = str(mat_id)
        return mats, mat_labels

    @property
    def sequences(self) -> Dict[int, Sequence]:
        """Return dictionary of sequences (load curves) keyed by ID.

        Sequence IDs are integers starting from 0.

        The first time this method is called for a FebReader object it creates and
        memoizes a dictionary of Sequence objects from the <loadcurve> elements in
        the FEBio XML.  Subsequent calls return the same Sequence objects.

        """
        fx = self.febioxml_module
        if self._sequences is None:
            self._sequences = fx.read_sequences(self.root)
        return self._sequences

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

        # Read named sets of geometric entities (node sets, element sets, face sets)
        # as *ordered lists*.  FEBio XML sometimes makes references into a set of by
        # positional index of its constituent entities, so the original XML order
        # must be preserved until we are done reading the XML file.
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
            if domain["material"] is None:
                continue
            nametype, name = domain["material"]
            material = model.named["materials"].obj(name, nametype)
            materials_used.add(material)
            for i in domain["elements"]:
                model.mesh.elements[i].material = material

        # Element data: material axes
        mat_axis_data = fx.read_elementdata_mat_axis(
            self.root, named_sets["element sets"]
        )
        for eset_name, eset_data in mat_axis_data.items():
            elementset = named_sets["element sets"][eset_name]
            for local_id, basis in eset_data:
                # local_id is the positional index in the element set list
                element_idx = elementset[local_id]
                # Who thought this much indirection in a data file was a good idea?
                mesh.elements[element_idx].basis = basis

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
            o_ids = ids_from_text(e_mcs_local[0].text)  # tuple
            if not np.all([ids_from_text(e.text) == o_ids for e in e_mcs_local]):
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
                ids = ids_from_text(e_mcs_local[0].text)  # 1-indexed
                for e in elements_by_mat[mat]:
                    e.basis = febioxml.basis_mat_axis_local(e, ids)

        # Read explicit rigid bodies.  Create a Body object for each
        # rigid body "material" in the XML with explicit geometry.
        explicit_bodies = {}
        for domain in domains:
            if domain["material"] is None:
                continue
            nametype, name = domain["material"]
            material = model.named["materials"].obj(name, nametype)
            ids = model.named["materials"].names(material, "ordinal_id")
            assert (len(ids)) == 1
            mat_id = ids[0]
            if isinstance(material, material_lib.Rigid):
                elements = []
                for i in domain["elements"]:
                    elements.append(model.mesh.elements[i])
                body = Body(elements)
                explicit_bodies[mat_id] = body
                # FEBio XML 4.0 uses names for rigid bodies
                explicit_bodies[name] = Body(elements)

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
            nodeset_name, mat_identifier = fx.read_rigid_interface(e_impbod)
            # TODO: Extract material resolution to a function if it needs to be used
            #  in multiple places
            if isinstance(mat_identifier, int):
                mat = model.named["materials"].obj(
                    mat_identifier, nametype="ordinal_id"
                )
            else:
                mat = model.named["materials"].obj(mat_identifier, nametype="canonical")
            mat_names = model.named["materials"].names(mat, "canonical")
            node_set = model.named["node sets"].obj(nodeset_name)
            if mat in materials_used:
                # This <rigid> element represents an explicit rigid body
                # ↔ node set interface.
                rigid_interface = RigidInterface(mat, node_set)
                model.constraints.append(rigid_interface)
            else:
                # This <rigid> element represents an implicit rigid body
                body = ImplicitBody(model.mesh, node_set, mat)
                implicit_bodies[mat_identifier] = body
                # FEBio XML 4.0 uses names for rigid bodies
                for k in mat_names:
                    implicit_bodies[k] = body

        # Read fixed boundary conditions.
        # TODO: Support solutes
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
        global_body_bcs = fx.read_body_bcs(
            self.root,
            explicit_bodies,
            implicit_bodies,
            model.named["sequences"]["ordinal_id"],
        )
        apply_body_bc(model, None, global_body_bcs)

        # Steps
        for e_step in self.root.findall(f"{fx.STEP_PARENT}/{fx.STEP_NAME}"):
            step, name = read_step(e_step, model, physics, self.febioxml_module)
            model.add_step(step, name)
            # Rigid body constraints.  So far, the <Step> tree mirrors the global
            # tree, so no need for version-specific code.
            step_body_bcs = fx.read_body_bcs(
                e_step,
                explicit_bodies,
                implicit_bodies,
                model.named["sequences"]["ordinal_id"],
            )
            apply_body_bc(model, step, step_body_bcs)
        # Prescribed nodal conditions.  Has to go after steps are created; otherwise
        # there are no steps to which to attach the applied conditions.
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

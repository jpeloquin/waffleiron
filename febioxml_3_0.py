# Base packages
from collections import defaultdict
from typing import Dict

# Same-package modules
from .core import ZeroIdxID, OneIdxID, Body, ImplicitBody, Extrapolant, Interpolant
from .control import SaveIters
from .febioxml import *

# These parts work the same as in FEBio XML 2.5
from .febioxml_2_5 import contact_bare_xml, meshdata_xml

# Facts about FEBio XML 3.0

# XML element parents and names
BODY_COND_PARENT = "Rigid"
BODY_COND_NAME = "rigid_constraint"
IMPBODY_PARENT = "Boundary"
IMPBODY_NAME = "bc[@type='rigid']"
MESH_PARENT = "Mesh"
ELEMENTDATA_PARENT = "MeshData"
NODEDATA_PARENT = "MeshData"
ELEMENTSET_PARENT = "Mesh"
NODESET_PARENT = "Mesh"
SURFACEPAIR_LEADER_NAME = "primary"
SURFACEPAIR_FOLLOWER_NAME = "secondary"
STEP_PARENT = "Step"
STEP_NAME = "step"

BC_TYPE_TAG: Dict[str, dict] = {
    "node": {"variable": "prescribe", "fixed": "fix"},
    "body": {
        ("variable", "displacement"): "prescribe",
        ("fixed", "displacement"): "fix",
        ("variable", "force"): "force",
    },
}

XML_RB_DOF_FROM_DOF = {
    "x1": "Rx",
    "x2": "Ry",
    "x3": "Rz",
    "α1": "Ru",
    "α2": "Rv",
    "α3": "Rw",
}
DOF_FROM_XML_RB_DOF = {v: k for k, v in XML_RB_DOF_FROM_DOF.items()}
VAR_FROM_XML_RB_DOF = {
    "Rx": "displacement",
    "Ry": "displacement",
    "Rz": "displacement",
    "Ru": "rotation",
    "Rv": "rotation",
    "Rw": "rotation",
}

# Map of Ticker fields → elements relative to <Step>
TICKER_PARAMS = {
    "n": ReqParameter("Control/time_steps", int),
    "dtnom": ReqParameter("Control/step_size", to_number),
    "dtmin": OptParameter(
        "Control/time_stepper/dtmin", to_number, 0
    ),  # Undocumented default, but zero makes sense as a minimum time
    # step, as no smaller value is allowed.
    "dtmax": OptParameter(
        "Control/time_stepper/dtmax", to_number, 1
    ),  # Undocumented default.  FEBio 3 uses 0 if the value is missing
    # (but doesn't enforce it for the first time step), which is
    # broken in multiple ways.  But if a load curve is provided the
    # scale appears to be ignored.  If a load curve is given without
    # a value (scale) is given, the default should probably be 1.  I
    # think FEBio 2 used 0.05 as the default.
}
# Map of Controller fields → elements relative to <Step>
CONTROLLER_PARAMS = {
    "max_retries": OptParameter("Control/time_stepper/max_retries", int, 5),
    "opt_iter": OptParameter("Control/time_stepper/opt_iter", int, 10),
    "save_iters": OptParameter("Control/plot_level", SaveIters, SaveIters.MAJOR),
}
# Map of Solver fields → elements relative to <Step>
SOLVER_PARAMS = {
    "dtol": OptParameter("Control/solver/dtol", to_number, 0.001),
    "etol": OptParameter("Control/solver/etol", to_number, 0.01),
    "rtol": OptParameter("Control/solver/rtol", to_number, 0),
    "lstol": OptParameter("Control/solver/lstol", to_number, 0.9),
    "ptol": OptParameter("Control/solver/ptol", to_number, 0.01),
    "min_residual": OptParameter("Control/solver/min_residual", to_number, 1e-20),
    "update_method": OptParameter("Control/solver/qnmethod", str, "BFGS"),
    "reform_each_time_step": OptParameter(
        "Control/solver/reform_each_time_step", text_to_bool, True
    ),
    "reform_on_diverge": OptParameter(
        "Control/solver/diverge_reform", text_to_bool, True
    ),
    "max_refs": OptParameter("Control/solver/max_refs", int, 15),
    "max_ups": OptParameter("Control/solver/max_ups", int, 10),
}

# Functions for reading FEBio XML 3.0


def iter_node_conditions(root):
    """Return generator over prescribed nodal condition info.

    Returns dict of property names → values.  All properties are
    not-None except the following:

    (1) "nodal values" will be None if the condition applies the same
    condition to all nodes.

    (2) "name" will be None if the boundary condition is not named.

    (3) "scale" will be None if the condition is heterogeneous, as FEBio
    XML 3.0 does not include a scale in this case.

    """
    step_id = -1  # Curent step ID (0-indexed)
    for e_Step in root.findall(f"{STEP_PARENT}/{STEP_NAME}"):
        step_id += 1
        for e_prescribe in e_Step.findall("Boundary/bc[@type='prescribe']"):
            # Re-initialize output
            info = {
                "name": None,
                "node set name": None,
                "axis": None,  # x1, fluid, charge, etc.
                "variable": None,  # displacement, force, pressure, etc.
                "sequence ID": None,
                "scale": None,  # not required in FEBio XML 3.0
                "relative": False,
                "nodal values": None,
                "step ID": None,
            }
            # Read values
            if "name" in e_prescribe.attrib:
                info["name"] = e_prescribe.attrib["name"]
            info["node set name"] = e_prescribe.attrib["node_set"]
            e_dof = find_unique_tag(e_prescribe, "dof")
            info["dof"] = DOF_NAME_FROM_XML_NODE_BC[e_dof.text]
            info["variable"] = VAR_FROM_XML_NODE_BC[e_dof.text]
            e_scale = find_unique_tag(e_prescribe, "scale")
            # ^ <scale> is required because it stores the load curve ID
            info["sequence ID"] = to_number(e_scale.attrib["lc"]) - 1
            if ("type" in e_scale.attrib) and (e_scale.attrib["type"] == "map"):
                # Heterogeneous condition
                e_NodeData = find_unique_tag(
                    root, f"{NODEDATA_PARENT}/NodeData[@name='{e_scale.text}']"
                )
                nm_nodeset = e_NodeData.attrib["node_set"]
                e_NodeSet = find_unique_tag(
                    root, f"{NODESET_PARENT}/NodeSet[@name='{nm_nodeset}']"
                )
                # The semantics of the lid attributes in <NodeData> are
                # unclear.  I am interpreting them as indices into the
                # <NodeSet> list.
                values = {}
                node_ids = [int(e.attrib["id"]) - 1 for e in e_NodeSet.findall("node")]
                info["nodal values"] = {
                    node_ids[int(e.attrib["lid"]) - 1]: to_number(e.text)
                    for e in e_NodeData.findall("node")
                }
            else:
                # Homogeneous condition
                info["scale"] = to_number(e_scale.text)
            e_relative = e_prescribe.find("relative")
            if e_relative is not None:
                info["relative"] = True
            info["step ID"] = step_id
            yield info


def read_domains(root: ET.Element):
    """Return list of domains"""
    domains = []
    e_domains = find_unique_tag(root, "MeshDomains")
    for e in e_domains:
        name = e.attrib["name"]
        material = e.attrib["mat"]
        e_domain = find_unique_tag(root, f"{MESH_PARENT}/Elements[@name='{name}']")
        elements = [
            ZeroIdxID(int(e.attrib["id"]) - 1) for e in e_domain.findall("elem")
        ]
        domain = {
            "name": name,
            "material": ("canonical", material),
            "elements": elements,
        }
        domains.append(domain)
    return domains


def apply_body_bc(model, e_rbc, explicit_bodies, implicit_bodies, step):
    """Read & apply a <rigid_constraint> element

    model := Model object.

    e_rigid_body := The <rigid_constraint> XML element.

    explicit_bodies := map of material ID → body.

    implicit_bodies := map of material ID → body.

    step := The step to which the rigid body boundary condition belongs.

    """
    # Each <rigid_body> element defines constraints for one rigid body,
    # identified by its material ID.  Constraints may be fixed
    # (atemporal) or time-varying (temporal).
    #
    # Get the Body object from the material id
    mat_id = int(find_unique_tag(e_rbc, "rb").text) - 1
    if mat_id in explicit_bodies:
        body = explicit_bodies[mat_id]
    else:
        # Assume mat_id refers to an implicit rigid body
        body = implicit_bodies[mat_id]
    # Variable displacement case:
    if e_rbc.attrib["type"] == BC_TYPE_TAG["body"][("variable", "displacement")]:
        var = "displacement"
        dof = DOF_FROM_XML_RB_DOF[find_unique_tag(e_rbc, "dof").text]
        e_seq = find_unique_tag(e_rbc, "value")
        seq = read_parameter(e_seq, model.named["sequences"])
        # Relative?
        e_relative = find_unique_tag(e_rbc, "relative")
        if e_relative is None:
            relative = False
        else:
            relative = text_to_bool(e_relative.text)
        model.apply_body_bc(body, dof, var, seq, relative=relative, step=step)
    # Fixed displacement case:
    elif e_rbc.attrib["type"] == BC_TYPE_TAG["body"][("fixed", "displacement")]:
        xml_dofs = (s.strip() for s in find_unique_tag(e_rbc, "dofs").text.split(","))
        for xml_dof in xml_dofs:
            dof = DOF_FROM_XML_RB_DOF[xml_dof]
            var = VAR_FROM_XML_RB_DOF[xml_dof]
            model.fixed["body"][(dof, var)].add(body)
    # TODO: variable force


def get_surface_name(surfacepair_subelement):
    """Return surface name for subelement of SurfacePair

    For example, return "surface1" for the element <primary>surface1</primary>.

    This function exists because the surface name was an attribute in FEBio XML 2.5.

    """
    return surfacepair_subelement.text


def read_fixed_node_bcs(root: Element, model):
    """Return nodesets with fixed degrees of freedom

    :param root: <febio_spec> Element
    :param nodesets: Map of nodeset name → nodeset.  All nodesets referenced by the
    fixed BC XML elements must have names stored in this parameter.
    :return: Map of (dof, var) → NodeSet

    In FEBio XML 3.0, the parent XML element is Boundary.  The fixed DoFs are stored
    in the text content of a child element <dofs> as a comma-separated string, like:

    <bc type="fix" set="nodeset_name">
      <dofs>x,y,z</dofs>
    </bc>

    """
    bcs = {}
    for e_fix in root.findall(f"Boundary/bc[@type='{BC_TYPE_TAG['node']['fixed']}']"):
        e_dofs = find_unique_tag(e_fix, "dofs")
        fx_kws = [kw.strip() for kw in e_dofs.text.split(",")]
        for k in fx_kws:
            dof = DOF_NAME_FROM_XML_NODE_BC[k]
            var = VAR_FROM_XML_NODE_BC[k]
            # In FEBio XML 3.0, the node set to which the fixed boundary condition is
            # applied is referenced by name.  The name must already be present in the
            # model's name registry.
            nodeset = model.named["node sets"].obj(e_fix.attrib["node_set"])
            bcs[(dof, var)] = nodeset
    return bcs


def parse_rigid_interface(e_bc):
    """Parse a <bc type="rigid"> element"""
    nodeset_name = e_bc.attrib["node_set"]
    e_rb = find_unique_tag(e_bc, "rb")
    mat_id = int(e_rb.text) - 1
    return nodeset_name, mat_id


def sequences(root: Element) -> Dict[int, Sequence]:
    """Return dictionary of sequence ID → sequence from FEBio XML 3.0"""
    sequences = {}
    for ord_id, e_lc in enumerate(root.findall("LoadData/load_controller")):
        if e_lc.attrib["type"] != "loadcurve":
            raise NotImplementedError(
                f"{e_lc.base}:{e_lc.sourceline} <load_controller> element of type '{e_lc.attrib['type']}' is not yet supported."
            )
        fake_id = int(e_lc.attrib["id"])
        e_points = find_unique_tag(e_lc, "points")
        curve = [read_point(e.text) for e in e_points]
        # Set extrapolation
        e_extend = find_unique_tag(e_lc, "extend")
        if e_extend is None:
            # FEBio default
            extrap = Extrapolant.CONSTANT
        else:
            extrap = EXTRAP_FROM_XML_EXTRAP[e_extend.text.lower()]
        # Set interpolation
        e_interp = find_unique_tag(e_lc, "interpolate")
        if e_interp is None:
            # FEBio default
            interp = Interpolant.LINEAR
        else:
            interp = INTERP_FROM_XML_INTERP[e_interp.text.lower()]
        # Create and store the Sequence object
        sequences[ord_id] = Sequence(
            curve, interp=interp, extrap=extrap, steplocal=False
        )
    return sequences


# Functions for writing FEBio XML 3.0


def body_constraints_xml(
    body, constraints: dict, material_registry, implicit_rb_mats, sequence_registry
):
    """Return <rigid_constraint> element(s) for body's constraints.

    The constrained variable can be displacement or rotation.

    """
    elems = []
    mat_id = body_mat_id(body, material_registry, implicit_rb_mats)
    # Can't put fixed and variable constraints in the same
    # <rigid_constraint> element, so first we have to group the
    # constraints by kind
    fixed_constraints = []
    variable_constraints = []
    for dof, bc in constraints.items():
        if bc["sequence"] == "fixed":
            fixed_constraints.append((dof, bc))
        else:  # bc['sequence'] is Sequence
            variable_constraints.append((dof, bc))
    # Create <rigid_constraint> element for fixed constraints
    if fixed_constraints:
        e_rb_fixed = ET.Element(BODY_COND_NAME)
        e_rb_fixed.attrib["type"] = BC_TYPE_TAG["body"][("fixed", "displacement")]
        ET.SubElement(e_rb_fixed, "rb").text = str(mat_id + 1)
        ET.SubElement(e_rb_fixed, "dofs").text = ",".join(
            XML_RB_DOF_FROM_DOF[dof] for dof, _ in fixed_constraints
        )
        elems.append(e_rb_fixed)
    # Create <rigid_constraint> element for variable constraints.  I
    # think you must use a separate element for each degree of freedom
    # (x, y, z, Rx, Ry, Rz).
    for dof, bc in variable_constraints:
        e_rb = ET.Element(BODY_COND_NAME)
        k = ("variable", bc["variable"])
        e_rb.attrib["type"] = BC_TYPE_TAG["body"][k]
        ET.SubElement(e_rb, "rb").text = str(mat_id + 1)
        ET.SubElement(e_rb, "dof").text = XML_RB_DOF_FROM_DOF[dof]
        seq = bc["sequence"]
        seq_id = get_or_create_seq_id(sequence_registry, seq)
        e_value = ET.SubElement(e_rb, "value")
        e_value.attrib["lc"] = str(seq_id + 1)
        v = bc["scale"]
        if isinstance(bc["sequence"], ScaledSequence):
            v = v * bc["sequence"].scale
        e_value.text = str(v)
        if bc["variable"] == "force":
            ET.SubElement(e_rb, "load_type").text = "0"
            # ^ semantics of this are unclear, but this is what FEBio
            # Studio 1.3 exports
        #
        # FEBio only supports relative constraints for displacement
        if bc["variable"] == "displacement":
            ET.SubElement(e_rb, "relative").text = bool_to_text(bc["relative"])
        elif bc["relative"]:
            # Most likely: bc['variable'] == "force"
            raise ValueError(
                f"FEBio XML does not permit relative {bc['variable']} conditions for bodies."
            )
        elems.append(e_rb)
    return elems


# contact_bare_xml is provided by febioxml_2_5


def mesh_xml(model, domains, material_registry):
    """Create <Mesh> and <MeshDomain> XML elements.

    <Geometry> became <Mesh> in FEBio XML 3.0

    """
    e_geometry = ET.Element(MESH_PARENT)
    e_meshdomains = ET.Element("MeshDomains")
    # Write <nodes>
    e_nodes = ET.SubElement(e_geometry, "Nodes")
    for i, x in enumerate(model.mesh.nodes):
        feb_nid = i + 1  # 1-indexed
        e = ET.SubElement(e_nodes, "node", id="{}".format(feb_nid))
        e.text = vec_to_text(x)
        e_nodes.append(e)
    # Write <Elements> and <SolidDomain> for each domain
    for i, domain in enumerate(domains):
        # <Elements>
        e_elements = ET.SubElement(e_geometry, "Elements", name=domain["name"])
        e_elements.attrib["type"] = domain["element_type"].feb_name
        for i, e in domain["elements"]:
            e_element = ET.SubElement(e_elements, "elem")
            e_element.attrib["id"] = str(i + 1)
            e_element.text = ", ".join(str(i + 1) for i in e.ids)
        # <SolidDomain>
        mat_name = material_registry.names(domain["material"], "canonical")[0]
        e_soliddomain = ET.SubElement(
            e_meshdomains, "SolidDomain", name=domain["name"], mat=mat_name
        )
    return e_geometry, e_meshdomains


def node_data_xml(nodes, data, data_name, nodeset_name):
    """Construct NodeData XML element"""
    e_NodeData = ET.Element("NodeData")
    e_NodeData.attrib["data_type"] = "scalar"
    # TODO: support other data types
    e_NodeData.attrib["name"] = data_name
    e_NodeData.attrib["node_set"] = nodeset_name
    # Write NodeData/node elements.  To specify a node, FEBio XML, for
    # some reason, uses the 1-indexed position of the node in the node
    # set as a "local ID", as opposed to just using the node's ID.  Our
    # node sets, being sets, are unordered.  To be able to generate the
    # local ID, we write NodeSet/node elements in FEBio XML in ascending
    # order of node ID.
    lid_from_node_id = {node_id: i + 1 for i, node_id in enumerate(sorted(nodes))}
    for i, v in zip(nodes, data):
        ET.SubElement(
            e_NodeData,
            "node",
            lid=str(lid_from_node_id[i]),
        ).text = float_to_text(v)
    return e_NodeData


def node_fix_disp_xml(fixed_conditions, nodeset_registry):
    """Return XML elements for node fixed displacement conditions.

    fixed_conditions := The data structure in model.fixed["node"]

    This function may create and add new nodesets to the nodeset name
    registry.  If generating a full XML tree, be sure to write these new
    nodesets to the tree.

    """
    # Tag hierarchy: <Boundary><bc type="fix" node_set="set_name">
    e_bcs = []
    # FEBio XML 3.0 stores the nodal BCs by node set, so we need to do
    # some collation first.
    by_nodeset = defaultdict(list)
    for dofvar, nodeset in fixed_conditions.items():
        if not nodeset:
            continue
        nodeset = NodeSet(nodeset)
        by_nodeset[nodeset].append(dofvar)
    for nodeset, dofvar_pairs in by_nodeset.items():
        # Get or create a name for the node set
        base = f"fixed_nodes_{','.join(dof for dof, var in dofvar_pairs)}_auto"
        nodeset_name = nodeset_registry.get_or_create_name(base, nodeset)
        # Create element
        e_bc = ET.Element(
            "bc", type=BC_TYPE_TAG["node"]["fixed"], node_set=nodeset_name
        )
        txt = ",".join(XML_BC_FROM_DOF[t] for t in dofvar_pairs)
        e_dofs = ET.SubElement(e_bc, "dofs").text = txt
        e_bcs.append(e_bc)
    return e_bcs


def node_var_disp_xml(
    model, xmlroot, nodes, scales, seq, dof, var, relative, step_name
):
    """Return XML elements for nodes variable displacement

    model := Model object.  Needed for the name registry.

    Returns tuple of (<bc> element, <NodeData> element)

    """
    # Hierarchy: <Boundary><bc type="prescribe" node_set="set_name">
    e_bc = ET.Element("bc", type=BC_TYPE_TAG["node"]["variable"])
    e_dof = ET.SubElement(e_bc, "dof").text = XML_BC_FROM_DOF[(dof, var)]
    seq_id = get_or_create_seq_id(model.named["sequences"], seq)
    e_sc = ET.SubElement(e_bc, "scale", lc=str(seq_id + 1), type="map")
    # Other subelements
    ET.SubElement(e_bc, "relative").text = str(int(relative))
    # Get or create a name for the node set
    nm_base = "nodal_bc_" f"step={step_name}_var={var[0]}_seq={seq_id}_autogen"
    nodeset = NodeSet(nodes)
    nodeset_name = model.named["node sets"].get_or_create_name(nm_base, nodeset)
    e_bc.attrib["node_set"] = nodeset_name
    # Generate a non-duplicate name for the Geometry/MeshData/NodeData
    # element, which will contain the node-specific scaling factors.
    stem = "nodal_bc_" f"step={step_name}_{dof}_seq{seq_id}_autogen"
    i = 0
    data_name = f"{stem}{i}"
    e_MeshData = find_unique_tag(xmlroot, "MeshData")
    while e_MeshData.find(f"NodeData[@name='{data_name}']") is not None:
        i += 1
        data_name = f"{stem}{i}"
    # Create the <NodeData> element
    e_NodeData = node_data_xml(nodes, scales, data_name, nodeset_name)
    # Reference the node-specific boundary condition scaling factors
    e_sc.text = data_name
    return e_bc, e_NodeData


def sequence_xml(sequence: Sequence, sequence_id: int, t0=0.0):
    """Return a <load_controller> XML element for a sequence.

    sequence := Sequence object.

    sequence_id := Integer ID (origin = 0) to use for the sequence's XML
    element "id" attribute.  The ID will be incremented by 1 to account
    for FEBio XML's use of 1-referenced IDs.

    t0 := Time offset to apply to the sequence's time points before
    writing them to XML.  The intended use for this is to translate from
    step-local to global simulation time.

    """
    e_loadcurve = ET.Element(
        "load_controller", id=str(sequence_id + 1), type="loadcurve"
    )
    ET.SubElement(e_loadcurve, "interpolate").text = XML_INTERP_FROM_INTERP[
        sequence.interpolant
    ]
    ET.SubElement(e_loadcurve, "extend").text = XML_EXTRAP_FROM_EXTRAP[
        sequence.extrapolant
    ]
    e_points = ET.SubElement(e_loadcurve, "points")
    for pt in sequence.points:
        ET.SubElement(e_points, "point").text = f"{pt[0] + t0}, {pt[1]}"
    return e_loadcurve


def surface_pair_xml(faceset_registry, primary, secondary, name):
    """Return SurfacePair XML element.

    The surfaces (face sets) involved in the surface pair must already have names in
    `faceset_registry`.

    """
    e_surfpair = ET.Element("SurfacePair", name=name)
    ET.SubElement(e_surfpair, "primary").text = faceset_registry.names(primary)[0]
    ET.SubElement(e_surfpair, "secondary").text = faceset_registry.names(secondary)[0]
    return e_surfpair


def step_xml_factory():
    """Create empty <step> XML elements"""
    i = 1
    while True:
        e = ET.Element(STEP_NAME, id=str(i), name=f"Step{i}")
        yield e

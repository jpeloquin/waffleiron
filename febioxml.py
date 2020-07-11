import os
import lxml.etree as ET
from .core import NodeSet, FaceSet, ElementSet
from .element import Quad4, Tri3, Hex8, Penta6, Element
from . import material
from .math import orthonormal_basis

# Map "bc" attribute value from <prescribe>, <prescribed>, <fix>, or
# <fixed> element to a variable name.  This list is valid for both node
# and rigid body conditions.  FEBio handles force conditions in other
# XML elements: for rigid bodies, <force>, and for nodes, <nodal_load>.
VAR_FROM_XML_NODE_BC = {'x': 'displacement',
                        'y': 'displacement',
                        'z': 'displacement',
                        'Rx': 'rotation',
                        'Ry': 'rotation',
                        'Rz': 'rotation',
                        'p': 'pressure'}
# Map "bc" attribute value from <prescribe>, <prescribed>,
# <fix>, or <fixed> element to a degree of freedom.
DOF_NAME_FROM_XML_NODE_BC = {"x": "x1",
                             "y": "x2",
                             "z": "x3",
                             "Rx": "α1",
                             "Ry": "α2",
                             "Rz": "α3",
                             "p": "fluid"}

TAG_FROM_BC = {'node': {'variable': 'prescribe',
                        'fixed': 'fix'},
               'body': {'variable': 'prescribed',
                        'fixed': 'fixed'}}


elem_cls_from_feb = {'quad4': Quad4,
                     'tri3': Tri3,
                     'hex8': Hex8,
                     'penta6': Penta6}

solid_class_from_name = {'isotropic elastic': material.IsotropicElastic,
                         'Holmes-Mow': material.HolmesMow,
                         'fiber-exp-pow': material.ExponentialFiber,
                         'fiber-pow-linear': material.PowerLinearFiber,
                         'neo-Hookean': material.NeoHookean,
                         'solid mixture': material.SolidMixture,
                         'rigid body': material.RigidBody,
                         'biphasic': material.PoroelasticSolid,
                         'Donnan equilibrium': material.DonnanSwelling,
                         'multigeneration': material.Multigeneration,
                         "orthotropic elastic": material.OrthotropicElastic}
solid_name_from_class = {v: k for k, v in solid_class_from_name.items()}

perm_class_from_name = {"perm-Holmes-Mow": material.IsotropicHolmesMowPermeability,
                        "perm-const-iso": material.IsotropicConstantPermeability}
perm_name_from_class = {v: k for k, v in perm_class_from_name.items()}


control_tagnames_to_febio = {'time steps': 'time_steps',
                             'step size': 'step_size',
                             'max refs': 'max_refs',
                             'max ups': 'max_ups',
                             'dtol': 'dtol',
                             'etol': 'etol',
                             'rtol': 'rtol',
                             'lstol': 'lstol',
                             'ptol': 'ptol',
                             # ^ for biphasic analysis
                             'plot level': 'plot_level',
                             'time stepper': 'time_stepper',
                             'max retries': 'max_retries',
                             'dtmax': 'dtmax',
                             'dtmin': 'dtmin',
                             'opt iter': 'opt_iter',
                             'min residual': 'min_residual',
                             'update method': 'qnmethod',
                             'symmetric biphasic': 'symmetric_biphasic',
                             # ^ for biphsaic analysis
                             'reform each time step': 'reform_each_time_step',
                             # ^ for fluid analysis
                             'reform on diverge': 'diverge_reform',
                             'analysis type': 'analysis'}
control_tagnames_from_febio = {v: k for k, v in control_tagnames_to_febio.items()}
control_values_to_febio = {'update method': {"quasi-Newton": "1",
                                             "BFGS": "0"}}
control_values_from_febio = {k: {v_xml: v_us for v_us, v_xml in conv.items()}
                             for k, conv in control_values_to_febio.items()}


# TODO: Redesign the compatibility system so that compatibility can be
# derived from the material's type.
module_compat_by_mat = {material.PoroelasticSolid: set(['biphasic']),
                        material.RigidBody: set(['solid', 'biphasic']),
                        material.OrthotropicElastic: set(['solid', 'biphasic']),
                        material.IsotropicElastic: set(['solid', 'biphasic']),
                        material.SolidMixture: set(['solid', 'biphasic']),
                        material.PowerLinearFiber: set(['solid', 'biphasic']),
                        material.ExponentialFiber: set(['solid', 'biphasic']),
                        material.HolmesMow: set(['solid', 'biphasic'])}


def _to_number(s):
    """Convert numeric string to int or float as appropriate."""
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

def bool_to_text(v):
    return "1" if v else "0"

def vec_to_text(v):
    return ', '.join(f"{a:.7e}" for a in v)


def bvec_to_text(v):
    return ', '.join(float_to_text(a) for a in v)


def float_to_text(a):
    return f"{a:.7g}"


def _find_unique_tag(root, path):
    """Find and return a tag or an error if > 1 of same."""
    tags = root.findall(path)
    if len(tags) == 1:
        return tags[0]
    elif len(tags) > 1:
        raise ValueError(f"Multiple `{path}` tags in file `{os.path.abspath(root.base)}`")


def read_named_sets(xml_root):
    """Read nodesets, etc., and apply them to a model."""
    sets = {'node sets': {},
            'face sets': {},
            'element sets': {}}
    tag_name = {'node sets': 'NodeSet',
                'face sets': 'Surface',
                'element sets': 'ElementSet'}
    cls_from_entity_type = {"node sets": NodeSet,
                            "face sets": FaceSet,
                            "element sets": ElementSet}
    # Handle items that are stored by id
    for k in ["node sets", "element sets"]:
        for e_set in xml_root.findall('./Geometry/' + tag_name[k]):
            cls = cls_from_entity_type[k]
            items = cls()
            for e_item in e_set.getchildren():
                item_id = int(e_item.attrib['id']) - 1
                items.update([item_id])
            sets[k][e_set.attrib['name']] = items
    # Handle items that are stored as themselves
    for k in ["face sets"]:
        for tag_set in xml_root.findall('./Geometry/' + tag_name[k]):
            cls = cls_from_entity_type[k]
            items = cls()
            for tag_item in tag_set.getchildren():
                items.append(_canonical_face([int(s.strip()) - 1
                                              for s in tag_item.text.split(",")]))
            sets[k][tag_set.attrib["name"]] = items
    return sets


def iter_node_conditions(root):
    """Return generator over prescribed nodal condition info.

    Returns dict of property names → values.  All properties are
    guaranteed to be not-None, except "nodal values", which will be None
    if the condition applies the same condition to all nodes.

    All returned IDs are 0-indexed for consistency with febtools.

    """
    step_id = -1  # Curent step ID (0-indexed)
    for e_Step in root.findall("Step"):
        step_id += 1
        for e_prescribe in e_Step.findall("Boundary/prescribe"):
            # Re-initialize output
            info = {"node set name": None,
                    "axis": None,  # x1, fluid, charge, etc.
                    "variable": None,  # displacement, force, etc.
                    "sequence ID": None,
                    "scale": 1.0,
                    "nodal values": None,
                    "step ID": None}
            # Read values
            info["node set name"] = e_prescribe.attrib["node_set"]
            info["axis"] = axis_from_febio[e_prescribe.attrib["bc"]]
            info["variable"] = "displacement"
            e_scale = e_prescribe.find("scale")
            seq_scale = 0.0  # FEBio default
            if e_scale.text is not None:
                seq_scale = _to_number(e_scale.text)
            info["sequence ID"] =  _to_number(e_scale.attrib["lc"]) - 1
            e_value = e_prescribe.find("value")
            if e_value is not None:
                if "node_data" in e_value.attrib:
                    # Node-specific data; look up the data in MeshData
                    info["scale"] = seq_scale
                    e_NodeSet = _find_unique_tag(root,
                                                 "Geometry/NodeSet[@name='" +
                                                 info['node set name'] +
                                                 "']")
                    e_NodeData = _find_unique_tag(root,
                                                  "MeshData/NodeData[@name='" +
                                                  e_value.attrib["node_data"] +
                                                  "']")
                    info["nodal values"] = {}
                    for e_node, e_value in zip(e_NodeSet.findall("node"),
                                               e_NodeData.findall("node")):
                        id_ = int(e_node.attrib["id"]) - 1
                        info["nodal values"][id_] = _to_number(e_value.text)
                else:
                    # One value for all nodes; redundant with "scale"
                    val_scale = _to_number(e_value.text)
                    info["scale"] = seq_scale * val_scale
            info["step ID"] = step_id
            yield info


def basis_mat_axis_local(element: Element,
                         local_ids=(1, 2, 4)):
    """Return element basis for FEBio XML <mat_axis type="local"> values.

    element is an Element object.

    mat_axis_local is a tuple of 3 element-local node IDs (1-indexed).
    The default value is (1, 2, 4) to match FEBio.  FEBio /treats/ (0,
    0, 0) as equal to (1, 2, 4), so this function does the same.

    """
    # FEBio special-case
    if local_ids == (0, 0, 0):
        local_ids = (1, 2, 4)
    a = element.nodes[local_ids[1] - 1] - element.nodes[local_ids[0] - 1]
    d = element.nodes[local_ids[2] - 1] - element.nodes[local_ids[0] - 1]
    basis = orthonormal_basis(a, d)
    return basis


def normalize_xml(root):
    """Convert some items in FEBio XML to 'normal' representation.

    FEBio XML allows some items to be specified several ways.  To reduce
    the complexity of the code that converts FEBio XML to a febtools
    Model, this function should be used ahead of time to normalize the
    representation of said items.

    Specific normalizations:

    - When a bare <Control> element exists, wrap it in a <Step> element.

    - When a bare <Boundary> element exists, wrap it in a <Step> element.

    - [TODO] Convert <mat_axis type="local">0,0,0</mat_axis> to the
      default value of 1,2,4.

    This function also does some validation.

    """
    # Validation: At most one of <Control> or <Step> should exist
    if root.find("Control") is not None and root.find("Step") is not None:
        msg = (f"{root.base} has both a <Control> and <Step> section. The FEBio documentation does not specify how these sections are supposed to interact, so normalization is aborted.")
        raise ValueError(msg)
    #
    # Normalization: When a bare <Control> element exists, wrap it in a
    # <Step> element.
    if root.find("Control") is not None:
        e_Control = root.find("Control")
        # From validation above, we know that no <Step> element exists,
        # so we need to create one.
        e_Step = ET.Element("Step")
        e_Control.getparent().remove(e_Control)
        root.insert(1, e_Step)
        e_Step.append(e_Control)
    #
    # Normalization: When a bare <Boundary> element exists, wrap any
    # <Boundary>/<prescribe> elements in the first <Step> element.
    e_rBoundary = root.find("Boundary")
    if e_rBoundary is not None:
        e_Step = root.find("Step")
        if e_Step is None:
            e_Step = ET.Element("Step")
            root.insert(1, e_Step)
        es_prescribe = e_rBoundary.findall("prescribe")
        # Do we need to create a Step/Boundary element?
        e_sBoundary = e_Step.find("Boundary")
        if len(es_prescribe) != 0 and e_sBoundary is None:
            e_sBoundary = ET.SubElement(e_Step, "Boundary")
        # Move the <prescribe> elements
        for e_prescribe in es_prescribe:
            e_rBoundary.remove(e_prescribe)
            e_sBoundary.append(e_prescribe)
        # Delete the <Boundary> element if it is now empty
        e_rBoundary = root.find("Boundary")
        if len(e_rBoundary) == 0:
            root.remove(e_rBoundary)
    return root

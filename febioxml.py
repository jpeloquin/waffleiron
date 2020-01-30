import lxml.etree as ET
from .core import NodeSet, FaceSet, ElementSet
from .element import Quad4, Tri3, Hex8, Penta6
from . import material

axis_to_febio = {'x1': 'x',
                 'x2': 'y',
                 'x3': 'z',
                 'α1': 'Rx',
                 'α2': 'Ry',
                 'α3': 'Rz',
                 'pressure': 'p'}

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
                         'multigeneration': material.Multigeneration}
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
                        material.LinearOrthotropicElastic: set(['solid', 'biphasic']),
                        material.IsotropicElastic: set(['solid', 'biphasic'])}


def vec_to_text(v):
    return ', '.join(f"{a:.7e}" for a in v)


def bvec_to_text(v):
    return ', '.join(f"{a:.7f}" for a in v)


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


def normalize_xml(root):
    """Convert some items in FEBio XML to 'normal' representation.

    FEBio XML allows some items to be specified several ways.  To reduce
    the complexity of the code that converts FEBio XML to a febtools
    Model, this function should be used ahead of time to normalize the
    representation of said items.

    Specific normalizations:

    - When a bare <Control> element exists, wrap it in a <Step> element.

    This function also does some validation.

    """
    # Validation: Only one of <Control> or <Step> should exist
    if root.find("Control") is None and root.find("Step") is None:
        msg = (f"{root.base} has both a <Control> and <Step> section. The FEBio documentation does not specify how these sections are supposed to interact, so normalization is aborted.")
        raise ValueError(msg)
    # Normalization: When a bare <Control> element exists, wrap it in a
    # <Step> element.
    if root.find("Control") is not None:
        e_Control = root.find("Control")
        e_Step = ET.Element("Step")
        e_Control.getparent().remove(e_Control)
        root.insert(1, e_Step)
        e_Step.append(e_Control)
    return root

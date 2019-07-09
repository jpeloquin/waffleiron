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

elem_cls_from_feb = {'quad4': Quad4,
                     'tri3': Tri3,
                     'hex8': Hex8,
                     'penta6': Penta6}

solid_class_from_name = {'isotropic elastic': material.IsotropicElastic,
                         'Holmes-Mow': material.HolmesMow,
                         'fiber-exp-pow': material.ExponentialFiber,
                         'neo-Hookean': material.NeoHookean,
                         'solid mixture': material.SolidMixture,
                         'rigid body': material.RigidBody,
                         'biphasic': material.PoroelasticSolid,
                         'Donnan equilibrium': material.DonnanSwelling}
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
                             'plot level': 'plot_level',
                             'time stepper': 'time_stepper',
                             'max retries': 'max_retries',
                             'dtmax': 'dtmax',
                             'dtmin': 'dtmin',
                             'opt iter': 'opt_iter'}
control_tagnames_from_febio = {v: k for k, v in control_tagnames_to_febio.items()}

# TODO: Redesign the compatibility system so that compatibility can be
# derived from the material's type.
module_compat_by_mat = {material.PoroelasticSolid: set(['biphasic']),
                        material.RigidBody: set(['solid', 'biphasic']),
                        material.LinearOrthotropicElastic: set(['solid', 'biphasic'])}

def read_named_sets(xml_root):
    """Read nodesets, etc., and apply them to a model."""
    sets = {'node sets': {},
            'facet sets': {},
            'element sets': {}}
    tag_name = {'node sets': 'NodeSet',
                'facet sets': 'Surface',
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
    for k in ["facet sets"]:
        for tag_set in xml_root.findall('./Geometry/' + tag_name[k]):
            cls = cls_from_entity_type[k]
            items = cls()
            for tag_item in tag_set.getchildren():
                items.append(tuple([int(s.strip()) - 1
                                    for s in tag_item.text.split(",")]))
            sets[k][tag_set.attrib["name"]] = items
    return sets

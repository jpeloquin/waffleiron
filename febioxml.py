from .element import Quad4, Tri3, Hex8

axis_to_febio = {'x1': 'x',
                 'x2': 'y',
                 'x3': 'z',
                 'pressure': 'p'}

elem_cls_from_feb = {'quad4': Quad4,
                     'tri3': Tri3,
                     'hex8': Hex8}

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

def read_named_sets(xml_root):
    """Read nodesets, etc., and apply them to a model."""
    sets = {'nodes': {},
            'facets': {},
            'elements': {}}
    set_list = (('nodes', 'NodeSet'),
                ('facets', 'Surface'),
                ('elements', 'ElementSet'))
    for k, tag in set_list:
        for e_set in xml_root.findall('./Geometry/' + tag):
            items = set([])
            for e_item in e_set.getchildren():
                i = int(e_item.attrib['id'])
                items.update([i])
            sets[k][e_set.attrib['name']] = items
    return sets

# Base packages
from math import degrees
from operator import itemgetter
# System packages
from lxml import etree as ET
# In-house packages
import febtools as feb

feb_version = 2.0

def exponentialfiber_to_feb(mat):
    """Convert ExponentialFiber material instance to FEBio xml.

    """
    e = ET.Element('material', type='fiber-exp-pow')
    p = ET.SubElement(e, 'alpha')
    p.text = str(mat.alpha)
    p = ET.SubElement(e, 'beta')
    p.text = str(mat.beta)
    p = ET.SubElement(e, 'ksi')
    p.text = str(mat.xi)
    p = ET.SubElement(e, 'theta')
    p.text = str(degrees(mat.theta))
    p = ET.SubElement(e, 'phi')
    p.text = str(degrees(mat.phi))
    return e

def holmesmow_to_feb(mat):
    """Convert HolmesMow material instance to FEBio xml.

    """
    e = ET.Element('material', type='Holmes-Mow')
    E, v = feb.material.fromlame(mat.y, mat.mu)
    ET.SubElement(e, 'E').text = str(E)
    p = ET.SubElement(e, 'v').text = str(v)
    ET.SubElement(e, 'beta').text = str(mat.beta)
    return e

def isotropicelastic_to_feb(mat):
    """Convert IsotropicElastic material instance to FEBio xml.

    """
    e = ET.Element('material', type='isotropic elastic')
    E, v = feb.material.fromlame(mat.y, mat.mu)
    ET.SubElement(e, 'E').text = str(E)
    ET.SubElement(e, 'v').text = str(v)
    return e

def solidmixture_to_feb(mat):
    """Convert SolidMixture material instance to FEBio xml.

    """
    e = ET.Element('material', type='solid mixture')
    for submat in mat.materials:
        m = material_to_feb(submat)
        m.tag = 'solid'
        e.append(m)
    return e

def material_to_feb(mat):
    """Convert a material instance to FEBio xml.

    """
    if mat is None:
        e = ET.Element('material', type='unknown')
    elif isinstance(mat, feb.material.ExponentialFiber):
        e = exponentialfiber_to_feb(mat)
    elif isinstance(mat, feb.material.HolmesMow):
        e = holmesmow_to_feb(mat)
    elif isinstance(mat, feb.material.IsotropicElastic):
        e = isotropicelastic_to_feb(mat)
    elif isinstance(mat, feb.material.SolidMixture):
        e = solidmixture_to_feb(mat)
    else:
        raise Exception("{} not implemented for conversion to FEBio xml.".format(mat.__class__))
    return e

def write_feb(model, fpath):
    """Write model to FEBio xml file.

    Inputs
    ------
    fpath : string
        Path for output file.

    materials : list of Material objects

    """
    root = ET.Element('febio_spec', version="{}".format(feb_version))
    Globals = ET.SubElement(root, 'Globals')
    Material = ET.SubElement(root, 'Material')
    Geometry = ET.SubElement(root, 'Geometry')
    Nodes = ET.SubElement(Geometry, 'Nodes')
    e_boundary = ET.SubElement(root, 'Boundary')
    e_loaddata = ET.SubElement(root, 'LoadData')
    Output = ET.SubElement(root, 'Output')

    # Typical MKS constants
    Constants = ET.SubElement(Globals, 'Constants')
    ET.SubElement(Constants, 'R').text = '8.314e-6'
    ET.SubElement(Constants, 'T').text = '294'
    ET.SubElement(Constants, 'Fc').text = '96485e-9'

    # Assign integer sequence ids.
    i = 0
    seq_id = {}
    for step in model.steps:
        # Sequences in nodal displacement boundary conditions
        for node_id, bc in step['bc'].iteritems():
            for axis, d in bc.iteritems():
                seq = d['sequence']
                if seq not in seq_id:
                    seq_id[seq] = i
                    i += 1
        # Sequences in dtmax
        if 'time stepper' in step['control']:
            if 'dtmax' in step['control']['time stepper']:
                dtmax = step['control']['time stepper']['dtmax']
                if dtmax.__class__ is feb.Sequence:
                    if dtmax not in seq_id:
                        seq_id[dtmax] = i
                        i += 1

    # Nodes section
    for i, x in enumerate(model.mesh.nodes):
        feb_nid = i + 1 # 1-indexed
        e = ET.SubElement(Nodes, 'node', id="{}".format(feb_nid))
        e.text = ",".join("{:e}".format(n) for n in x)
        Nodes.append(e)

    # Materials section
    # enumerate materials
    material_ids = {k:v for v, k in enumerate(set(e.material
                        for e in model.mesh.elements))}

    # make material tags
    # sort by id to get around FEBio bug
    materials = [(i, mat) for mat, i in material_ids.iteritems()]
    materials.sort(key=itemgetter(1))
    for i, m in materials:
        tag = feb.output.material_to_feb(m)
        tag.attrib['name'] = 'Material' + str(i + 1)
        tag.attrib['id'] = str(i + 1)
        Material.append(tag)

    # Elements section
    # assemble elements into blocks with like type and material
    elemsets= {}
    for i, elem in enumerate(model.mesh.elements):
        typ = elem.__class__.__name__.lower()
        mid = material_ids[elem.material]
        elemsets.setdefault(elem.material, {}).setdefault(elem.__class__, []).append(elem)
    # write element sets
    for mat, d in elemsets.iteritems():
        for typ, elems in d.iteritems():
            e_elements = ET.SubElement(Geometry, 'Elements',
                mat=str(material_ids[elem.material] + 1),
                type=typ.__name__.lower())
            for i, elem in enumerate(elems):
                e_elem = ET.SubElement(e_elements, 'elem',
                                       id=str(i + 1))
                e_elem.text = ','.join(str(i + 1) for i in elem.ids)

        # # write thicknesses for shells
        # if label == 'tri3' or label == 'quad4':
        #     ElementData = root.find('Geometry/ElementData')
        #     if ElementData is None:
        #         ElementData = ET.SubElement(Geometry, 'ElementData')
        #     e = ET.SubElement(ElementData, 'element', id=str(feb_eid))
        #     t = ET.SubElement(e, 'thickness')
        #     t.text = '0.001, 0.001, 0.001'

    # Boundary section (fixed nodal BCs)
    for axis, nodeset in model.fixed_nodes.iteritems():
        if nodeset:
            e_fix = ET.SubElement(e_boundary, 'fix', bc=axis)
            for nid in nodeset:
                ET.SubElement(e_fix, 'node', id=str(nid + 1))

    # LoadData (load curves)
    # sort sequences by id to get around FEBio bug
    sequences = [(i, seq) for seq, i in seq_id.iteritems()]
    sequences.sort(key=itemgetter(1))
    for i, seq in sequences:
        e_lc = ET.SubElement(e_loaddata, 'loadcurve', id=str(i+1),
                             type=seq.typ, extend=seq.extend)
        for pt in seq.points:
            ET.SubElement(e_lc, 'point').text = ','.join(str(x) for x in pt)

    # Output section
    plotfile = ET.SubElement(Output, 'plotfile', type='febio')
    ET.SubElement(plotfile, 'var', type='displacement')
    ET.SubElement(plotfile, 'var', type='stress')

    # Step section(s)
    for i, step in enumerate(model.steps):
        e_step = ET.SubElement(root, 'Step',
                               name='Step{}'.format(i + 1))
        e_module = ET.SubElement(e_step, 'Module',
                                 type=step['module'])
        e_con = ET.SubElement(e_step, 'Control')
        ET.SubElement(e_con, 'analysis',
                      type=step['control']['analysis type'])
        tbl = {'time steps': 'time_steps',
               'step size': 'step_size',
               'max refs': 'max_refs',
               'max ups': 'max_ups',
               'dtol': 'dtol',
               'etol': 'etol',
               'rtol': 'rtol',
               'lstol': 'lstol',
               'plot level': 'plot_level'}
        for lbl1, lbl2 in tbl.iteritems():
            ET.SubElement(e_con, lbl2).text = \
                str(step['control'][lbl1])
        e_ts = ET.SubElement(e_con, 'time_stepper')
        ET.SubElement(e_ts, 'dtmin').text = \
            str(step['control']['time stepper']['dtmin'])
        ET.SubElement(e_ts, 'max_retries').text = \
            str(step['control']['time stepper']['max retries'])
        ET.SubElement(e_ts, 'opt_iter').text = \
            str(step['control']['time stepper']['opt iter'])
        # dtmax may have an associated sequence
        dtmax = step['control']['time stepper']['dtmax']
        e_dtmax = ET.SubElement(e_ts, 'dtmax')
        if dtmax.__class__ is feb.Sequence:
            e_dtmax.attrib['lc'] = str(seq_id[dtmax] + 1)
            e_dtmax.text = "1"
        else:
            e_dtmax.text = str(dtmax)

        # Boundary conditions
        e_bd = ET.SubElement(e_step, 'Boundary')
        # collect BCs into FEBio-like data structure
        prescribed = {}
        for i, ax_bc in step['bc'].iteritems():
            for ax, d in ax_bc.iteritems():
                v = d['value']
                seq = d['sequence']
                prescribed.setdefault(seq, {}).setdefault(ax, {})[i] = v
        # write out data
        for seq, d in prescribed.iteritems():
            for axis, vnodes in d.iteritems():
                e_pres = ET.SubElement(e_bd, 'prescribe',
                                       bc=str(axis),
                                       lc=str(seq_id[seq] + 1))
                for nid, v in vnodes.iteritems():
                    e_node = ET.SubElement(e_pres, 'node', id=str(nid + 1)).text = str(v)

    tree = ET.ElementTree(root)
    with open(fpath, 'w') as f:
        tree.write(f, pretty_print=True, xml_declaration=True,
                   encoding='us-ascii')

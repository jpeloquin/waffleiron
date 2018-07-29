# Base packages
from collections import defaultdict
from math import degrees
# System packages
from lxml import etree as ET
# In-house packages
import febtools as feb
from .conditions import Sequence
from .control import step_duration
from .febioxml import control_tagnames_to_febio

feb_version = 2.0

axis_to_febio = {'x1': 'x',
                 'x2': 'y',
                 'x3': 'z',
                 'pressure': 'p'}


def exponentialfiber_to_feb(mat):
    """Convert ExponentialFiber material instance to FEBio XML.

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
    """Convert HolmesMow material instance to FEBio XML.

    """
    e = ET.Element('material', type='Holmes-Mow')
    E, v = feb.material.fromlame(mat.y, mat.mu)
    ET.SubElement(e, 'E').text = str(E)
    ET.SubElement(e, 'v').text = str(v)
    ET.SubElement(e, 'beta').text = str(mat.beta)
    return e


def isotropicelastic_to_feb(mat):
    """Convert IsotropicElastic material instance to FEBio XML.

    """
    e = ET.Element('material', type='isotropic elastic')
    E, v = feb.material.fromlame(mat.y, mat.mu)
    ET.SubElement(e, 'E').text = str(E)
    ET.SubElement(e, 'v').text = str(v)
    return e


def linear_orthotropic_elastic_to_feb(mat):
    """Convert LinearOrthotropicElastic material instance to FEBio XML.

    """
    e = ET.Element('material', type='orthotropic elastic')
    # Material properties
    ET.SubElement(e, 'E1').text = str(mat.E1)
    ET.SubElement(e, 'E2').text = str(mat.E2)
    ET.SubElement(e, 'E3').text = str(mat.E3)
    ET.SubElement(e, 'G12').text = str(mat.G12)
    ET.SubElement(e, 'G23').text = str(mat.G23)
    ET.SubElement(e, 'G31').text = str(mat.G31)
    ET.SubElement(e, 'v12').text = str(mat.v12)
    ET.SubElement(e, 'v23').text = str(mat.v23)
    ET.SubElement(e, 'v31').text = str(mat.v31)
    # Symmetry axes
    axes = ET.SubElement(e, 'mat_axis', type='vector')
    ET.SubElement(axes, 'a').text = ','.join([str(a) for a in mat.x1])
    ET.SubElement(axes, 'd').text = ','.join([str(a) for a in mat.x2])
    return e


def neo_hookean_to_feb(mat):
    """Convert NeoHookean material instance to FEBio XML.

    """
    e = ET.Element('material', type='neo-Hookean')
    E, v = feb.material.fromlame(mat.y, mat.mu)
    ET.SubElement(e, 'E').text = str(E)
    ET.SubElement(e, 'v').text = str(v)
    return e


def poroelastic_to_feb(mat):
    """Convert Poroelastic material instance to FEBio XML.

    """
    e = ET.Element('material', type='biphasic')
    # Add solid material
    m = material_to_feb(mat.solid_material)
    m.tag = 'solid'
    e.append(m)
    # Add permeability
    txt_from_kind = {'constant isotropic': 'perm-const-iso'}
    e_permeability = ET.SubElement(e, 'permeability',
                                   type=txt_from_kind[mat.kind])
    ET.SubElement(e_permeability, 'perm').text = str(mat.permeability)
    return e


def solidmixture_to_feb(mat):
    """Convert SolidMixture material instance to FEBio XML.

    """
    e = ET.Element('material', type='solid mixture')
    for submat in mat.materials:
        m = material_to_feb(submat)
        m.tag = 'solid'
        e.append(m)
    return e


def material_to_feb(mat):
    """Convert a material instance to FEBio XML.

    """
    if mat is None:
        e = ET.Element('material', type='unknown')
    else:
        f = {feb.material.ExponentialFiber: exponentialfiber_to_feb,
             feb.material.HolmesMow: holmesmow_to_feb,
             feb.material.IsotropicElastic: isotropicelastic_to_feb,
             feb.material.NeoHookean: neo_hookean_to_feb,
             feb.material.LinearOrthotropicElastic: linear_orthotropic_elastic_to_feb,
             feb.material.PoroelasticSolid: poroelastic_to_feb,
             feb.material.SolidMixture: solidmixture_to_feb}
        try:
            e = f[type(mat)](mat)
        except ValueError:
            msg = "{} not implemented for conversion to FEBio XML."
            print(msg.format(mat.__class__))
            raise
    return e


def xml(model):
    """Convert a model to an FEBio XML tree.

    This is meant to allow XML-editing trickery, if necessary, prior to
    writing the XML to an on-disk .feb .

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
        for node_id, bc in step['bc'].items():
            for axis, d in bc.items():
                if 'sequence' in d:
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
        feb_nid = i + 1  # 1-indexed
        e = ET.SubElement(Nodes, 'node', id="{}".format(feb_nid))
        e.text = ",".join("{:e}".format(n) for n in x)
        Nodes.append(e)

    # Materials section
    # enumerate materials
    material_ids = {k: v for v, k
                    in enumerate(set(e.material for e in model.mesh.elements))}

    # make material tags
    # sort by id to get around FEBio bug
    materials = [(i, mat) for mat, i in material_ids.items()]
    materials.sort()
    for i, m in materials:
        tag = feb.output.material_to_feb(m)
        try:
            tag.attrib['name'] = model.material_labels[i]
        except KeyError:
            pass
        tag.attrib['id'] = str(i + 1)
        Material.append(tag)

    # Elements and ElementData sections

    # Assemble elements into blocks with like type and material.
    # Elemsets uses material instances as keys.  Each item is a
    # dictionary using element classes as keys,
    # with items being tuples of (element_id, element).
    elemsets = {}
    for i, elem in enumerate(model.mesh.elements):
        subdict = elemsets.setdefault(elem.material, {})
        like_elements = subdict.setdefault(elem.__class__, [])
        like_elements.append((i, elem))

    # write element sets
    e_elementdata = ET.SubElement(Geometry, 'ElementData')
    for mat, d in elemsets.items():
        for ecls, like_elems in d.items():
            e_elements = ET.SubElement(Geometry, 'Elements',
                                       mat=str(material_ids[mat] + 1),
                                       type=ecls.__name__.lower())
            for i, e in like_elems:
                # write the element's node ids
                e_elem = ET.SubElement(e_elements, 'elem',
                                       id=str(i + 1))
                e_elem.text = ','.join(str(i + 1) for i in e.ids)
                # write any defined element data
                tagged = False
                # ^ track if an element tag has been created in
                # ElementData for the current element
                if 'thickness' in elem.properties:
                    if not tagged:
                        e_edata = ET.SubElement(e_elementdata, 'element',
                                                id=str(i + 1))
                        tagged = True
                    ET.SubElement(e_edata, 'thickness').text = \
                        ','.join(str(t) for t in elem.properties['thickness'])
                if 'v_fiber' in elem.properties:
                    if not tagged:
                        e_edata = ET.SubElement(e_elementdata, 'element',
                                                id=str(i + 1))
                        tagged = True
                    ET.SubElement(e_edata, 'fiber').text = \
                        ','.join(str(a) for a in elem.properties['v_fiber'])

    # FEBio needs Nodes to precede Elements to precede ElementData.
    # It apparently has very limited xml parsing.
    geo_subs = {'Nodes': [],
                'Elements': [],
                'ElementData': []}
    for e in Geometry:
        geo_subs[e.tag].append(e)
    Geometry[:] = geo_subs['Nodes'] + geo_subs['Elements']
    # Only add optional tags if they contain data.  Otherwise FEBio
    # chokes and gives an error message incorrectly blaming the
    # following line in the XML file.
    if e_elementdata[:] != []:
        Geometry[:] += geo_subs['ElementData']

    # Boundary section (fixed nodal BCs)
    for axis, nodeset in model.fixed_nodes.items():
        if nodeset:
            e_fix = ET.SubElement(e_boundary, 'fix', bc=axis_to_febio[axis])
            for nid in nodeset:
                ET.SubElement(e_fix, 'node', id=str(nid + 1))

    # LoadData (load curves)
    # sort sequences by id to get around FEBio bug
    sequences = [(i, seq) for seq, i in seq_id.items()]
    sequences.sort()
    # Apply offset to load curves so they start at the same time the
    # previous step ends (global time), as required by FEBio.  In
    # `febtools`, each step has its own running time (local time).
    cumulative_time = 0.0
    for step in model.steps:
        # Adjust must point curves
        dtmax = step['control']['time stepper']['dtmax']
        if isinstance(dtmax, Sequence):
            dtmax.points = [(cumulative_time + t, v) for t, v in dtmax.points]
        # Adjust variable boundary condition curves
        curves_to_adjust = set([])
        for i, ax_bc in step['bc'].items():
            for ax, d in ax_bc.items():
                if not d == 'fixed':  # varying ("prescribed") BC
                    curves_to_adjust.update([d['sequence']])
        for curve in curves_to_adjust:
            curve.points = [(cumulative_time + t, v)
                            for t, v in curve.points]
        # Tally running time
        duration = step_duration(step)
        cumulative_time += duration
    # Write adjusted sequences
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
    cumulative_time = 0.0
    for i, step in enumerate(model.steps):
        e_step = ET.SubElement(root, 'Step',
                               name='Step{}'.format(i + 1))
        ET.SubElement(e_step, 'Module',
                      type=step['module'])
        # TODO: Warn if there's a poroelastic material and a solid
        # analysis is requested.  Or set the appropriate module
        # automatically.
        e_con = ET.SubElement(e_step, 'Control')
        ET.SubElement(e_con, 'analysis',
                      type=step['control']['analysis type'])
        for lbl1, lbl2 in control_tagnames_to_febio.items():
            if lbl1 in step['control'] and lbl1 != 'time stepper':
                txt = str(step['control'][lbl1])
                ET.SubElement(e_con, lbl2).text = txt
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
        if isinstance(dtmax, Sequence):
            # Reference the load curve for dtmax
            e_dtmax.attrib['lc'] = str(seq_id[dtmax] + 1)
            e_dtmax.text = "1"
        else:
            e_dtmax.text = str(dtmax)

        # Boundary conditions
        e_bd = ET.SubElement(e_step, 'Boundary')
        # collect BCs into FEBio-like data structure
        varying = defaultdict(dict)
        fixed = defaultdict(set)
        for i, ax_bc in step['bc'].items():
            for ax, d in ax_bc.items():
                if d == 'fixed':  # fixed BC
                    fixed[ax].add(i)
                else:  # varying ("prescribed") BC
                    v = d['value']
                    seq = d['sequence']
                    varying[seq].setdefault(ax, {})[i] = v
        # Write varying nodal BCs
        for seq, d in varying.items():
            for axis, vnodes in d.items():
                e_pres = ET.SubElement(e_bd, 'prescribe',
                                       bc=axis_to_febio[axis],
                                       lc=str(seq_id[seq] + 1))
                for nid, v in vnodes.items():
                    e_node = ET.SubElement(e_pres, 'node', id=str(nid + 1)).text = str(v)
        # Write fixed nodal BCs
        for axis, node_ids in fixed.items():
            e_axis = ET.SubElement(e_bd, 'fix', bc=axis_to_febio[axis])
            for i in node_ids:
                ET.SubElement(e_axis, 'node', id=str(i + 1))

    tree = ET.ElementTree(root)
    return tree


def write_xml(tree, f):
    """Write an XML tree to a .feb file"""
    tree.write(f, pretty_print=True, xml_declaration=True,
               encoding='utf-8')


def write_feb(model, f):
    """Write model's FEBio XML representation to a file object.

    Inputs
    ------
    fpath : string
        Path for output file.

    materials : list of Material objects

    """
    tree = xml(model)
    write_xml(tree, f)

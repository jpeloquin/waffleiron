# Base packages
from collections import defaultdict
# System packages
from lxml import etree as ET
# Same-package modules
from .output import material_to_feb
from .conditions import Sequence
from .control import step_duration
from .febioxml import control_tagnames_to_febio, axis_to_febio

feb_version = 2.0

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
                if dtmax.__class__ is Sequence:
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
        tag = material_to_feb(m)
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


def split_bc_names(s):
    """Split boundary condition names.

    In FEBio XML 2.0, each BC is one character.

    """
    return [c for c in s]

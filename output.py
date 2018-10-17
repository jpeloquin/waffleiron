# Base packages
from collections import defaultdict
from math import degrees
# System packages
from lxml import etree as ET
# Within-module packages
import febtools as feb
from .core import Body, ContactConstraint
from .conditions import Sequence
from .control import step_duration
from . import febioxml_2_5 as febioxml
# ^ The intent here is to eventually be able to switch between FEBio XML
# formats by exchanging this import statement for a different version.
# Common functionality can be shared between febioxml_*_*.py files via
# imports.

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


def rigid_body_to_feb(mat):
    """Convert SolidMixture material instance to FEBio XML.

    """
    e = ET.Element('material', type='rigid body')
    if mat.density is None:
        density = 1
    else:
        density = mat.density
    ET.SubElement(e, 'density').text = str(density)
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
             feb.material.SolidMixture: solidmixture_to_feb,
             feb.material.RigidBody: rigid_body_to_feb}
        try:
            e = f[type(mat)](mat)
        except ValueError:
            msg = "{} not implemented for conversion to FEBio XML."
            print(msg.format(mat.__class__))
            raise
    return e


def xml(model, version='2.5'):
    """Convert a model to an FEBio XML tree.

    Creating an FEBio XML tree from a model is useful because it allows
    XML-editing trickery, if necessary, prior to writing the XML to an
    on-disk .feb file.

    """
    root = ET.Element('febio_spec', version="{}".format(version))
    Globals = ET.SubElement(root, 'Globals')
    Material = ET.SubElement(root, 'Material')

    # Enumerate materials.  We do this early because in FEBio XML the
    # material ids are needed to define the geometry and meshdata
    # sections.
    material_ids = {k: v for v, k
                    in enumerate(set(e.material for e in model.mesh.elements))}

    parts = febioxml.parts(model)
    Geometry = febioxml.geometry_section(model, parts, material_ids)
    root.append(Geometry)

    e_boundary = ET.SubElement(root, 'Boundary')
    e_constraints = ET.SubElement(root, 'Constraints')
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
        for obj in step['bc']:
            for k in step['bc'][obj]:
                for ax, v in step['bc'][obj][k].items():
                    if type(v) is dict and 'sequence' in v:
                        seq = v['sequence']
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

    # Materials section
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

    # Permanently fixed nodes
    for axis, nodeset in model.fixed['node'].items():
        if nodeset:
            e_fix = ET.SubElement(e_boundary, 'fix', bc=febioxml.axis_to_febio[axis])
            for nid in nodeset:
                ET.SubElement(e_fix, 'node', id=str(nid + 1))
    # Permanently fixed bodies
    body_bcs = {}
    for axis, bodies in model.fixed['body'].items():
        for body in bodies:
            body_bcs.setdefault(body, set()).add(axis)
    for body, axes in body_bcs.items():
        mat_id = material_ids[body.elements[0].material]
        # TODO: ensure that body is all one material
        e_body = ET.SubElement(e_constraints, 'rigid_body',
                               mat=str(mat_id + 1))
        for ax in axes:
            ET.SubElement(e_body, 'fixed', bc=febioxml.axis_to_febio[ax])

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
        for i, ax_bc in step['bc']['node'].items():
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

    e_contact = contact_section(model)
    root.append(e_contact)
    # The <Contact> tag must come before the <Step> tag or FEBio will
    # only apply the specified contact constraints to the last step
    # (this is an FEBio bug).

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
        for lbl1, lbl2 in febioxml.control_tagnames_to_febio.items():
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
        #
        # FEBio XML spreads the boundary conditions (constraints) out in
        # difficult-to-organize way.
        #
        # For nodal contraints, there is one parent tag per kind + axis
        # + sequence, and one child tag per node + value.  The parent
        # tag may be named 'prescribe' or 'fix'.
        #
        # For body constraints, there is one parent tag per body, and
        # one child tag per kind + axis + sequence + value.  The parent
        # tag may be named 'prescribed' or 'fixed'.  (So much for
        # consistency.)
        #
        # FEBio does seem to handle empty tags appropriately, which
        # helps.
        tag_memo = {'node': {},
                    'body': {}}
        # ^ keeps track of which XML tags we've created already; access
        # as tag_memo[obj][kind][ax][seq], where `kind` ∈ {'fixed',
        # 'variable'}
        bc_tag_nm = {'node': {'variable': 'prescribe',
                              'fixed': 'fix'},
                     'body': {'variable': 'prescribed',
                              'fixed': 'fixed'}}
        e_nodal = ET.SubElement(e_step, 'Boundary')
        e_body = ET.SubElement(e_step, 'Constraints')
        for obj in step['bc']:  # node or body
            for k in step['bc'][obj]:  # node id or Body instance
                for ax in step['bc'][obj][k]:  # axis
                    v = step['bc'][obj][k][ax]
                    if type(v) is str and v == 'fixed':
                        kind = 'fixed'
                    elif type(v) is dict:
                        kind = 'variable'
                        seq = v['sequence']
                        v = v['value']
                    if obj == 'node':
                        e_grandfather = e_nodal
                        e_parent = tag_memo[obj] \
                            .setdefault(kind, {}) \
                            .setdefault(ax, {}) \
                            .setdefault(seq, ET.SubElement(e_grandfather,
                                                           bc_tag_nm[obj][kind],
                                                           bc=febioxml.axis_to_febio[ax],
                                                           lc=str(seq_id[seq] + 1)))
                        e_child = ET.SubElement(e_parent, 'node', id=str(k + 1))
                        if kind == 'variable':
                            e_child.text = str(v)
                    elif obj == 'body':
                        e_grandfather = e_body
                        mat_id = material_ids[k.elements[0].material]
                        e_parent = tag_memo[obj] \
                            .setdefault(k, ET.SubElement(e_grandfather, 'rigid_body',
                                                         mat=str(mat_id + 1)))
                        e_child = ET.SubElement(e_parent, bc_tag_nm[obj][kind],
                                                bc=febioxml.axis_to_febio[ax],
                                                lc=str(seq_id[seq] + 1))
                        if kind == 'variable':
                            e_child.text = str(v)

    tree = ET.ElementTree(root)
    return tree


def contact_section(model):
    tag_branch = ET.Element('Contact')
    contact_constraints = [constraint for constraint in model.constraints
                           if type(constraint) is ContactConstraint]
    for contact in contact_constraints:
        tag_contact = ET.SubElement(tag_branch, 'contact', type=contact.algorithm)
        # Write penalty-related tags
        ET.SubElement(tag_contact, 'auto_penalty') \
          .text = "1" if contact.penalty['type'] == 'auto' else "0"
        ET.SubElement(tag_contact, 'penalty').text = f"{contact.penalty['factor']}"
        # Write algorithm modification tags
        ET.SubElement(tag_contact, 'laugon').text = "1" if contact.augmented_lagrange else "0"
        # (two_pass would go here)
        # Write surfaces
        e_master = ET.SubElement(tag_contact, 'surface', type="master")
        for f in contact.master:
            e_master.append(tag_face(f))
        e_follower = ET.SubElement(tag_contact, 'surface', type="slave")
        for f in contact.follower:
            e_follower.append(tag_face(f))
    return tag_branch

def tag_face(face):
    nm = {3: "tri3",
          4: "quad4"}
    tag = ET.Element(nm[len(face)])
    tag.text = ", ".join([f"{i+1}" for i in face])
    return tag

def write_xml(tree, f):
    """Write an XML tree to a .feb file"""
    tree.write(f, pretty_print=True, xml_declaration=True,
               encoding='utf-8')


def write_feb(model, f, version='2.5'):
    """Write model's FEBio XML representation to a file object.

    Inputs
    ------
    fpath : string
        Path for output file.

    materials : list of Material objects

    """
    tree = xml(model, version=version)
    write_xml(tree, f)

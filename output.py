# Base packages
from collections import defaultdict
from math import degrees
# System packages
from lxml import etree as ET
# Within-module packages
import febtools as feb
from .core import Body, ImplicitBody, ContactConstraint
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


def add_autogen_nodeset(model, xml_root, name, nodes):
    """Autogenerate and add a nodeset to FEBio XML.

    This function is especially useful for autogenerating nodesets for
    FEBio 2.5 boundary conditions.

    """
    nm_base = name
    i = 0
    while name in model.named_sets:
        name = nm_base + "_" + str(i)
    model.named_sets['nodes'][name] = nodes
    # Create a tag for each node
    e_geometry = xml_root.find("./Geometry")
    e_nodeset = ET.SubElement(e_geometry, "NodeSet", name=name)
    for node_id in nodes:
        ET.SubElement(e_nodeset, 'node', id=str(node_id + 1))
    return name


def choose_module(materials):
    """Determine which module should be used to run the model.

    Currently only chooses between solid and biphasic.

    """
    module = "solid"
    for m in materials:
        if isinstance(m, feb.material.PoroelasticSolid):
            module = "biphasic"
    return module

def xml(model, version='2.5'):
    """Convert a model to an FEBio XML tree.

    Creating an FEBio XML tree from a model is useful because it allows
    XML-editing trickery, if necessary, prior to writing the XML to an
    on-disk .feb file.

    """
    # Enumerate all materials assigned to elements.  We do this early
    # because in FEBio XML the material ids are needed to define the
    # geometry and meshdata sections.  Technically, implicit bodies also
    # have a rigid material, but because the geometry is implicit they
    # can be added to the end of the material section as they are
    # encountered when handling boundary conditions.
    material_ids_by_material = {k: v for v, k in
                                enumerate(set(e.material for e in
                                              model.mesh.elements))}

    root = ET.Element('febio_spec', version="{}".format(version))
    Globals = ET.SubElement(root, 'Globals')
    Material = ET.SubElement(root, 'Material')

    root = ET.Element('febio_spec', version="{}".format(version))
    version_major, version_minor = [int(a) for a in version.split(".")]
    if version_major == 2 and version_minor >= 5:
        e_module = ET.SubElement(root, 'Module')
        # <Module> must exist and be first tag for FEBio XML ≥ 2.5.  We
        # need to figure out what the module should be, but will do that
        # later once the materials are enumerated.
        module = choose_module([m for m in material_ids_by_material])
        e_module.attrib["type"] = module

    Globals = ET.SubElement(root, 'Globals')
    Material = ET.SubElement(root, 'Material')

    parts = febioxml.parts(model)
    Geometry = febioxml.geometry_section(model, parts, material_ids_by_material)
    root.append(Geometry)

    e_boundary = ET.SubElement(root, 'Boundary')
    e_contact = contact_section(model)
    root.append(e_contact)
    # The <Contact> tag must come before the first <Step> tag or FEBio
    # will only apply the specified contact constraints to the last step
    # (this is an FEBio bug).
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
                    # v['sequence'] is Sequence object or 'fixed'
                    if type(v['sequence']) is Sequence:
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
    #
    # Make material tags for all materials assigned to elements.  Sort
    # by id to get around FEBio bug (FEBio ignores the ID attribute and
    # just uses tag order).
    materials = [(i, mat) for mat, i in material_ids_by_material.items()]
    materials.sort()
    for i, m in materials:
        tag = material_to_feb(m)
        if 'name' in tag.attrib:
            tag.attrib['name'] = model.material_labels[mat]
        tag.attrib['id'] = str(i + 1)
        Material.append(tag)
    # Assemble a list of all implicit rigid bodies used in the model.
    # There is currently no list of rigid bodies in the model or mesh
    # objects, so we have to search for them.  Rigid bodies may be
    # referenced in fixed constraints (model.fixed['body'][k] where k ∈
    # 'x1', 'x2', 'x3', 'α1', 'α2', 'α3') or in
    # model.steps[i]['bc']['body'] for each step i.
    implicit_bodies_to_process = set()  # memo
    # Search fixed constraints for rigid bodies
    for k in model.fixed['body']:
        for body in model.fixed['body'][k]:
            if isinstance(body, ImplicitBody):
                implicit_bodies_to_process.add(body)
    # Search steps' constraints for rigid bodies
    for step in model.steps:
        for body in step['bc']['body']:
            if isinstance(body, ImplicitBody):
                implicit_bodies_to_process.add(body)
    # Create FEBio rigid materials for all implicit rigid bodies and add
    # their rigid interfaces with the mesh.  That the implicit material
    # is rigid is an assumption, but an implicit deformable material in
    # FEA wouldn't make any sense.
    implicit_rigid_material_by_body = {}
    for i, implicit_body in enumerate(implicit_bodies_to_process):
        body_name = f"implicit_rigid_body_{i+1}"
        # Create the implicit body's FEBio rigid material
        mat = feb.material.RigidBody()
        tag = material_to_feb(mat)
        # TODO: Support comments in reader
        # tag.append(ET.Comment("Implicit rigid body"))
        mat_id = len(Material)
        tag.attrib['id'] = str(mat_id + 1)
        tag.attrib['name'] = body_name + "_psuedo-material"
        Material.append(tag)
        # Update material registries
        material_ids_by_material[mat] = mat_id
        implicit_rigid_material_by_body[body] = mat
        #
        # Add the implicit body's rigid interface with the mesh
        if version == '2.0':
            # FEBio XML 2.0 puts rigid bodies under §Constraints
            e_interface = ET.SubElement(e_contact, 'contact',
                                        type='rigid')
            for i in implicit_body.interface:
                # assumes interface is node list
                ET.SubElement(e_interface, 'node', id=str(i + 1),
                              rb=str(mat_id + 1))
        elif version_major == 2 and version_minor >= 5:
            # FEBio XML 2.0 puts rigid bodies under §Boundary
            name_base = f"{body_name}_interface"
            # assumes interface is node list
            name = add_autogen_nodeset(model, root, name_base, implicit_body.interface)
            ET.SubElement(e_boundary, "rigid", rb=str(mat_id + 1),
                          node_set=name)

    # Permanently fixed nodes
    for axis, nodeset in model.fixed['node'].items():
        if nodeset:
            if version == '2.0':
                # Tag heirarchy: <Boundary><fix bc="x"><node id="1"> for each node
                e_fixed_nodeset = ET.SubElement(e_boundary, 'fix',
                                                bc=febioxml.axis_to_febio[axis])
                for i in nodeset:
                    ET.SubElement(e_fixed_nodeset, 'node', id=str(i + 1))
            elif version_major == 2 and version_minor >= 5:
                # Tag heirarchy: <Boundary><fix bc="x" node_set="set_name">
                nm_base = f"fixed_{axis}_autogen-nodeset"
                nm = add_autogen_nodeset(model, root, nm_base, nodeset)
                # Create the tag
                ET.SubElement(e_boundary, 'fix', bc=febioxml.axis_to_febio[axis],
                              node_set=nm)
    # Permanently fixed bodies
    body_bcs = {}
    # Choose where to put rigid body constraints depending on FEBio XML
    # version.
    if version == '2.0':
        e_bc_body_parent = e_constraints
    elif version_major == 2 and version_minor >= 5:
        e_bc_body_parent = e_boundary
    # Collect rigid body boundary conditionsn in a more convenient
    # heirarchy
    for axis, bodies in model.fixed['body'].items():
        for body in bodies:
            body_bcs.setdefault(body, set()).add(axis)
    # Create the tags specifying fixed constraints for the rigid bodies
    for body, axes in body_bcs.items():
        e_body = ET.SubElement(e_bc_body_parent, 'rigid_body')
        # Assign the body's material
        if isinstance(body, Body):
            body_material = body.elements[0].material
            # TODO: ensure that body is all one material
        elif isinstance(body, ImplicitBody):
            body_material = implicit_rigid_material_by_body[body]
        mat_id = material_ids_by_material[body_material]
        e_body.attrib['mat'] = str(mat_id + 1)
        # Write tags for each of the fixed degrees of freedom
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
        # Gather must point curves
        dtmax = step['control']['time stepper']['dtmax']
        if isinstance(dtmax, Sequence):
            dtmax.points = [(cumulative_time + t, v) for t, v in dtmax.points]
        # Gather variable boundary condition / constraint curves
        curves_to_adjust = set([])
        for i, ax_bc in step['bc']['node'].items():
            for ax, d in ax_bc.items():
                if d == 'variable':  # varying ("prescribed") BC
                    curves_to_adjust.add(d['sequence'])
        # Gather the body constraint curves
        for body, body_constraints in step['bc']['body'].items():
            for ax, params in body_constraints.items():
                # params = {'variable': variable <string>,
                #           'sequence': Sequence object or 'fixed',
                #           'scale': scale <numeric>
                if type(params['sequence']) is Sequence:
                    curves_to_adjust.add(params['sequence'])
                    # TODO: Add test to exercise this code
        # Adjust the curves
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
    visited_implicit_bodies = set()
    for i, step in enumerate(model.steps):
        e_step = ET.SubElement(root, 'Step',
                               name='Step{}'.format(i + 1))
        if version == '2.0':
            ET.SubElement(e_step, 'Module', type=step['module'])
        # Warn if there's an incompatibility between requested materials
        # and modules.
        for mat in material_ids_by_material:
            if step['module'] not in febioxml.module_compat_by_mat[type(mat)]:
                raise ValueError(f"Material `{type(mat)}` is not compatible with Module {step['module']}")
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
        # amongst many tags, in a rather disorganized fashion.
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
        bc_tag_nm = {'node': {'variable': 'prescribe',
                              'fixed': 'fix'},
                     'body': {'variable': 'prescribed',
                              'fixed': 'fixed'}}
        e_bc_nodal_parent = ET.SubElement(e_step, 'Boundary')
        if version == '2.0':
            e_bc_body_parent = ET.SubElement(e_step, 'Constraints')
        elif version_major == 2 and version_minor >= 5:
            e_bc_body_parent = e_bc_nodal_parent
        # Collect nodal BCs in a more convenient heirarchy for writing FEBio XML
        node_memo = {}  # node_memo['fixed'|'variable'][axis] =
                        # {'nodes': [], 'scales': []}
        for node_id in step['bc']['node']:
            for ax in step['bc']['node'][node_id]:  # axis
                bc = step['bc']['node'][node_id][ax]
                if bc['sequence'] == 'fixed':
                    kind = 'fixed'
                else:  # bc['sequence'] is Sequence
                    kind = 'variable'
                node_memo.setdefault(kind, {}).setdefault(ax, {})['kind'] = kind
                node_memo[kind][ax]['sequence'] = bc['sequence']
                node_memo[kind][ax].setdefault('nodes', []).append(node_id)
                node_memo[kind][ax].setdefault('scales', []).append(bc['scale'])
        # TODO: support kind == 'fixed'.  (Does that make sense for a step?)
        for kind in node_memo:
            for axis in node_memo[kind]:
                bc = node_memo[kind][axis]
                e_bc = ET.SubElement(e_bc_nodal_parent,
                                     bc_tag_nm['node'][kind],  # 'fix' | 'prescribe'
                                     bc=febioxml.axis_to_febio[axis])
                if version == '2.0':
                    if kind == 'variable':
                        e_bc.attrib['lc'] =  str(seq_id[bc['sequence']] + 1)
                    # Write nodes as children of <Step><Boundary><prescribe>
                    for i, sc in zip(bc['nodes'], bc['scales']):
                        ET.SubElement(e_bc, 'node', id=str(i+1)).text = f"{sc:.7e}"
                elif version_major == 2 and version_minor >= 5:
                    # Use <Step><Boundary><prescribe node_set="set_name">
                    #
                    # Test if the node-specific scaling factors are all
                    # equal; if they are not, the BC cannot be
                    # represented as a single node set in FEBio XML 2.5.
                    sc0 = bc['scales'][0]
                    if all([sc == sc0 for sc in bc['scales']]):
                        # All scaling factors are equal
                        nm_base = f"step{i+1}_{kind}_{axis}_autogen-nodeset"
                        name = add_autogen_nodeset(model, root, nm_base, bc['nodes'])
                    else:
                        msg = (f"A nodal boundary condition was defined with "
                               "non-equal node-specific scaling factors and FEBio XML "
                               "{version} was requested, but FEBio XML {version} can "
                               "only support a single scaling factor for the entire "
                               "node set.")
                        # TODO: Add support for node-specific BCs using
                        # MeshData/NodeData.
                        raise ValueError(msg)
                    if kind == 'variable':
                        e_sc = ET.SubElement(e_bc, 'scale',
                                             lc=str(seq_id[bc['sequence']] + 1))
                        e_sc.text = f"{sc0:.7e}"
                        ET.SubElement(e_bc, 'relative').text = "0"
                    e_bc.attrib['node_set'] = name
        for body in step['bc']['body']:
            # Create or find the associated materials
            if isinstance(body, Body):
                # If an explicit body, its elements define its
                # materials.  We assume that the body is homogenous.
                mat = body.elements[0].material
                mat_id = material_ids_by_material[mat]
            elif isinstance(body, ImplicitBody):
                mat = implicit_rigid_material_by_body[body]
                mat_id = material_ids_by_material[mat]
            else:
                msg = f"body {k} does not have a supported type.  " + \
                    "Supported body types are Body and ImplicitBody."
                raise ValueError(msg)
            # Create the XML tags for the rigid body BC
            e_body = ET.SubElement(e_bc_body_parent, 'rigid_body', mat=str(mat_id + 1))
            for axis in step['bc']['body'][body]:
                bc = step['bc']['body'][body][ax]
                if bc['sequence'] == 'fixed':
                    kind = 'fixed'
                else:  # bc['sequence'] is Sequence
                    kind = 'variable'
                    seq = bc['sequence']
                    v = bc['scale']
                # Determine which tag name to use for the specified
                # variable: force or displacement
                if bc['variable'] == 'displacement':
                     tagname = bc_tag_nm['body'][kind]
                elif bc['variable'] == 'force':
                     tagname = 'force'
                else:
                     raise ValueError(f"Variable {bc['variable']} not supported for BCs.")
                e_bc = ET.SubElement(e_body, tagname,
                                     bc=febioxml.axis_to_febio[axis])
                if kind == 'variable':
                    e_bc.attrib['lc'] = str(seq_id[seq] + 1)
                    e_bc.text = str(v)

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
        for f in contact.leader:
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

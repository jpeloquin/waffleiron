# Base packages
from collections import defaultdict
# System packages
from lxml import etree as ET
# Same-package modules
from .core import Sequence
from .control import step_duration
from .febioxml import *

feb_version = 2.5

def parts(model):
    """Return list of parts.

    Here, 1 part = all elements of the same type with the same material.

    PreView has a notion of parts that is separate from both material
    and element type and named element sets.  This doesn't seem to
    matter to FEBio itself, though.  It may interfere with
    postprocessing, in which case the definition of a "part" from the
    perspective of febtools will have to be changed.

    """
    # TODO: Modify the definition of parts such that 1 part = all
    # *connected* elements with the same material.
    #
    # Assemble elements into blocks with like type and material.
    # Elemsets uses material instances as keys.  Each item is a
    # dictionary using element classes as keys, with items being tuples
    # of (element_id, element).
    by_mat_type = {}
    for i, elem in enumerate(model.mesh.elements):
        subdict = by_mat_type.setdefault(elem.material, {})
        like_elements = subdict.setdefault(elem.__class__, [])
        like_elements.append((i, elem))
    # Convert nested dictionaries to a list
    parts = []
    for mat in by_mat_type:
        for typ in by_mat_type[mat]:
            parts.append({'label': None,
                          'material': mat,
                          'element_type': typ,
                          'elements': by_mat_type[mat][typ]})
    return parts

def geometry_section(model, parts, material_registry):
    """Return XML tree for Geometry section."""
    e_geometry = ET.Element('Geometry')
    # Write <nodes>
    e_nodes = ET.SubElement(e_geometry, 'Nodes')
    for i, x in enumerate(model.mesh.nodes):
        feb_nid = i + 1  # 1-indexed
        e = ET.SubElement(e_nodes, 'node', id="{}".format(feb_nid))
        e.text = vec_to_text(x)
        e_nodes.append(e)
    # Write <elements> for each part
    for part in parts:
        e_elements = ET.SubElement(e_geometry, 'Elements')
        e_elements.attrib['type'] = part['element_type'].feb_name
        mat_id = material_registry.name(part['material'], "ordinal_id")
        e_elements.attrib['mat'] = str(mat_id + 1)
        for i, e in part['elements']:
            e_element = ET.SubElement(e_elements, 'elem')
            e_element.attrib['id'] = str(i + 1)
            e_element.text = ', '.join(str(i+1) for i in e.ids)
    return e_geometry

def vec_to_text(v):
    return ', '.join(f"{a:.7e}" for a in v)

def elem_var_fiber_xml(e):
    tag = ET.Element('elem')
    raise NotImplementedError
    # TODO: Implement this.  But it is not clear how the fiber direction
    # element property is supposed to be written in FEBio XML 2.5.
    # PreView won't export it.

def elem_var_thickness_xml(e):
    raise NotImplementedError

def elem_var_mataxis_xml(e):
    tag = ET.Element('elem')
    for nm, v in zip(['a', 'd'], e.properties['mat_axis']):
        ET.SubElement(tag, nm).text = vec_to_text(v)
    return tag

def elem_var_vonmises_xml(e):
    raise NotImplementedError

def elem_var_prestretch_xml(e):
    raise NotImplementedError

element_var_feb = {'v_fiber': {'name': 'fiber',
                               'fn': elem_var_fiber_xml},
                   'thickness': {'name': 'shell thickness',
                                 'fn': elem_var_thickness_xml},
                   'mat_axis': {'name': 'mat_axis',
                                'fn': elem_var_mataxis_xml},
                   'von Mises': {'name': 'MRVonMisesParameters',
                                 'fn': elem_var_vonmises_xml},
                   'prestretch': {'name': 'pre_stretch',
                                  'fn': elem_var_prestretch_xml}}

def meshdata_section(model, parts):
    """Return XML tree for MeshData section."""
    e_meshdata = ET.Element('MeshData')
    # Element data
    for part in parts:
        # Gather a list of elements for each variable
        e_by_var = {}
        for i, e in part['elements']:
            for p in e.properties:
                e_by_var.setdefault(p, [])
                e_by_var[p].append((i, e))
        # Write the element properties for each element in this part to XML.
        for var in e_by_var:
            # Only add optional tags if they contain data.  Otherwise
            # FEBio dies and incorrectly blames the following line in
            # the XML file.
            if len(e_by_var) != 0:
                e_elementdata = ET.SubElement(e_meshdata, 'ElementData',
                                              var=element_var_feb[var]['name'],
                                              elem_set=part['label'])
                for i, e in e_by_var[var]:
                    e_val = element_var_feb[var]['fn'](e)
                    e_val.attrib['lid'] = str(i + 1)
                    e_elementdata.append(e_val)
    return e_meshdata


def split_bc_names(s):
    """Split boundary condition names.

    In FEBio XML 2.5, each boundary condition is separated by a comma.

    """
    return [bc.strip() for bc in s.split(",")]

# import xml.etree.ElemOAentTree as ET
from lxml import etree as ET
import febtools as feb


feb_version = 1.2

def feb_skeleton():
    root = ET.Element('febio_spec', version="{}".format(feb_version))
    Globals = ET.SubElement(root, 'Globals')
    Material = ET.SubElement(root, 'Material')
    Geometry = ET.SubElement(root, 'Geometry')
    Nodes = ET.SubElement(Geometry, 'Nodes')
    Elements = ET.SubElement(Geometry, 'Elements')
    Boundary = ET.SubElement(root, 'Boundary')
    LoadData = ET.SubElement(root, 'LoadData')
    Output = ET.SubElement(root, 'Output')

    # Typical MKS constants
    Constants = ET.SubElement(Globals, 'Constants')
    c = {'R': '8.314e-6',
         'T': '298',
         'Fc': '96485e-9'}
    for k, v in c.iteritems():
        e = ET.SubElement(Constants, k)
        e.text = v

    return root

def solidmixture_to_feb(mat):
    """Convert SolidMixture material instance to FEBio xml.

    """
    e = ET.Element('material', type='solid mixture')
    for submat in mat.materials:
        m = material_to_feb(submat)
        m.tag = 'solid'
        e.append(m)
    return e

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
    p.text = str(mat.theta)
    p = ET.SubElement(e, 'phi')
    p.text = str(mat.phi)
    return e

def holmesmow_to_feb(mat):
    """Convert HolmesMow material instance to FEBio xml.

    """
    e = ET.Element('material', type='Holmes-Mow')
    E, v = feb.material.fromlame(mat.y, mat.mu)
    p = ET.SubElement(e, 'E')
    p.text = str(E)
    p = ET.SubElement(e, 'v')
    p.text = str(v)
    p = ET.SubElement(e, 'beta')
    p.text = str(mat.beta)
    return e

def material_to_feb(mat):
    """Convert a material instance to FEBio xml.

    """
    if isinstance(mat, feb.material.ExponentialFiber):
        e = exponentialfiber_to_feb(mat)
    elif isinstance(mat, feb.material.HolmesMow):
        e = holmesmow_to_feb(mat)
    elif isinstance(mat, feb.material.SolidMixture):
        e = solidmixture_to_feb(mat)
    else:
        raise Exception("{} not implemented for conversion to FEBio xml.".format(mat.__class__))
    return e

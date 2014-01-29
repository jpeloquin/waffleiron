# import xml.etree.ElemOAentTree as ET
from lxml import etree as ET


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

    

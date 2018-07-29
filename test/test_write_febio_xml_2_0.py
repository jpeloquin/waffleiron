import os
import febtools as feb
import febtools.febioxml_2_0 as febioxml_2_0

mesh = feb.mesh.hexa.rectangular_prism(1, 1, 1, 0.5)
model = feb.Model(mesh)
xml = febioxml_2_0.xml(model)
pth = os.path.splitext(os.path.basename(__file__))[0] + '.feb'
with open(pth, 'wb') as f:
    feb.output.write_xml(xml, f)
os.remove(pth)

from math import radians
import numpy as np
import febtools as feb
from febtools.control import auto_control_section
from febtools.core import NodeSet

if __name__ == "__main__":
    x1 = np.array([0, 0, 0])
    x2 = [np.cos(radians(17))*np.cos(radians(6)),
          np.sin(radians(17))*np.cos(radians(6)),
          np.sin(radians(6))]
    x3 = np.array([0.8348, 0.9758, 0.3460])
    x4 = np.array([0.0794, 0.9076, 0.1564])
    x5 = x1 + np.array([0.638*np.cos(radians(26))*np.sin(radians(1)),
                        0.638*np.sin(radians(26))*np.sin(radians(1)),
                        np.cos(radians(1))])
    x6 = x5 + np.array([0.71*np.cos(radians(-24))*np.cos(radians(-7)),
                        0.71*np.sin(radians(-24))*np.cos(radians(-7)),
                        np.sin(radians(-7))])
    x7 = [1, 1, 1]
    x8 = x5 + [np.sin(radians(9))*np.cos(radians(-11)),
               np.cos(radians(9))*np.cos(radians(-11)),
               np.sin(radians(-11))]
    nodes = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])
    element = feb.element.Hex8.from_ids([i for i in range(8)], nodes)
    material = feb.material.LinearOrthotropicElastic({"E1": 23,
                                                      "E2": 81,
                                                      "E3": 50,
                                                      "G12": 15,
                                                      "G23": 82,
                                                      "G31": 5,
                                                      "ν12": 0.26,
                                                      "ν23": -0.36,
                                                      "ν31": 0.2})
    element.material = material
    model = feb.Model(feb.Mesh(nodes, [element]))
    sequence = feb.Sequence(((0, 0), (1, 1)),
                            extend="extrapolate", typ="linear")
    model.add_step(control=auto_control_section(sequence, pts_per_segment=1))
    F = np.array([[1.34, 0.18, -0.11],
                  [-0.20, 1.14, 0.17],
                  [-0.11, 0.20, 0.93]])
    feb.conditions.prescribe_deformation(model,
                                         NodeSet([i for i in range(len(nodes))]),
                                         F,
                                         sequence)
    tree = feb.output.xml(model)
    # Add F tensor logfile output so we can check result manually
    e_Output = tree.find("Output")
    e_logfile = e_Output.makeelement("logfile")
    e_elementdata = e_logfile.makeelement("element_data",
                                          file="mat_axis_local_-_element_data.txt",
                                          data="Fxx;Fyy;Fzz;Fxy;Fyz;Fxz;Fyx;Fzy;Fzx",
                                          format="%i %g %g %g %g %g %g %g %g %g")
    e_logfile.append(e_elementdata)
    e_Output.append(e_logfile)
    with open("mat_axis_local.feb", "wb") as f:
        feb.output.write_xml(tree, f)

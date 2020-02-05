# Python built-ins
from math import radians
import os
# Public packages
import numpy as np
# febtools' local modules
import febtools as feb
from febtools.material import fromlame


def gen_model_center_crack_Hex8():
    """A 10 mm × 20 mm rectangle with a center 2 mm crack.

    Material: isotropic linear elastic.

    Boundary conditions: 2% strain applied in y.

    """
    reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'center_crack_uniax_isotropic_elastic_hex8.feb'))
    model = reader.model()
    soln = feb.input.XpltReader(os.path.join('test', 'fixtures', 'center_crack_uniax_isotropic_elastic_hex8.xplt'))
    model.apply_solution(soln)

    material = model.mesh.elements[0].material
    γ = material.y
    μ = material.mu
    E, ν = fromlame(γ, μ)

    crack_line = ((-0.001, 0.0, 0.0),
                       ( 0.001, 0.0, 0.0))

    # right tip
    tip_line_r = [i for i, (x, y, z)
                  in enumerate(model.mesh.nodes)
                  if (np.allclose(x, crack_line[1][0])
                      and np.allclose(y, crack_line[1][1]))]
    tip_line_r = set(tip_line_r)

    tip_line_l = [i for i, (x, y, z)
                  in enumerate(model.mesh.nodes)
                  if (np.allclose(x, crack_line[0][0])
                      and np.allclose(y, crack_line[0][1]))]
    tip_line_l = set(tip_line_l)

    # identify crack faces
    f_candidates = feb.selection.surface_faces(model.mesh)
    f_seed = [f for f in f_candidates
              if (len(set(f) & tip_line_r) > 1)]
    f_crack_surf = feb.selection.f_grow_to_edge(f_seed,
                                                model.mesh)
    crack_faces = f_crack_surf

    attrib = {"E": E,
              "ν": ν,
              "tip_line_l": tip_line_l,
              "tip_line_r": tip_line_r,
              "crack_line": crack_line,
              "crack_faces": crack_faces}

    return model, attrib


def gen_model_single_spiky_Hex8(material=None):
    """Return a model consisting of a single spiky Hex8 element.

    None of the edges of the Hex8 element are parallel to each other.
    The element is intended to be used as a fixture in tests of
    element-local basis vectors or issues related to spatial variation
    in shape function interpolation within the element.

    Material: Isotropic linear elastic.

    Boundary conditions: None.

    """
    # Create an irregularly shaped element in which no two edges are
    # parallel.
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
    element.material = material
    model = feb.Model(feb.Mesh(nodes, [element]))
    return model

import os
import numpy as np

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

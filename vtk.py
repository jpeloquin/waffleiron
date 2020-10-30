import numpy as np
from tvtk.api import tvtk
from .element import Hex8

# Mapping from febtools cell types to vtk cell types
vtk_cell_type = {Hex8: tvtk.Hexahedron().cell_type}


def tvtk_ugrid_from_mesh(mesh):
    """Create tvtk UnstructuredGrid object from mesh."""
    points = np.array(mesh.nodes)
    cells = np.array([[len(e.ids)] + e.ids for e in mesh.elements])
    cells = cells.ravel()
    # ^ when creating unstructured grid, a cell row starts with the
    # number of points in the cell
    cell_types = [vtk_cell_type[type(e)] for e in mesh.elements]
    cell_types = np.array(cell_types)
    cell_offsets = [len(e.ids) + i for i, e in enumerate(mesh.elements)]
    vtk_cells = tvtk.CellArray()
    vtk_cells.set_cells(len(cells), cells)

    ugrid = tvtk.UnstructuredGrid(points=points)

    # ugrid.set_cells(cell_types[0], cells)
    # ^ this works for meshes with one type

    ugrid.set_cells(cell_types, cell_offsets, vtk_cells)
    # ^ need to give cell offsets if cell_types is an array rather
    # than a scalar, but the cell offsets don't seem to matter;
    # Paraview and Mayavi still work fine even with obviously wrong
    # values (e.g. all 0)

    return ugrid


def tvtk_ugrid_from_model(model):
    """Create a tvtk UnstructuredGrid object from model.

    This function is similar to tvtk_ugrid_from_mesh.  The only
    difference is that it assigns data to the grid if data exists in
    the model.

    """
    ugrid = tvtk_ugrid_from_mesh(model.mesh)
    pass

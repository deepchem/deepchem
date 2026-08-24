"""
Structured mesh generator.

``generate_unit_square_mesh(nx, ny)`` creates an (nx+1) x (ny+1) regular
grid, splits each quad into two triangles, and auto-tags boundary nodes.
"""

import numpy as np

from .mesh import FEMMesh


def generate_unit_square_mesh(nx, ny):
    xs = np.linspace(0, 1, nx + 1, dtype=np.float32)  # nx is the number of divisions along the x axis beetween 0 and 1
    ys = np.linspace(0, 1, ny + 1, dtype=np.float32)  # ny is the number of divisions along the y axis beetween 0 and 1
    # es nx=4, ny=4 means a 4×4 grid of a square of side of lengh 1
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # two (xx and yy) 2D arrays of shape (nx+1, ny+1)
    # with the first index i runs over xs and the second index j runs over ys
    nodes = np.stack([xx.ravel(), yy.ravel()], axis=1)  # .ravel() flattens the 2D arrays

    # buildind the triangular elements by looping over the cells. Each cells can be individuate by
    # n0 =  n0 = i * (ny + 1) + j ; n1 = (i + 1) * (ny + 1) + j ; n2 = (i + 1) * (ny + 1) + (j + 1)
    # and n3 = i * (ny + 1) + (j + 1). For reference:
    #    n3 -- n2
    #    |  /  |
    #   n0 -- n1
    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + (j + 1)
            n3 = i * (ny + 1) + (j + 1)

            # split the cell into two triangles...
            elements.append([n0, n1, n2])  # the lower right trianlge
            elements.append([n0, n2, n3])  # the upper left triangle
    elements = np.array(elements, dtype=np.int64)  # converts the list into a numpy array

    idx = lambda i, j: i * (ny + 1) + j  # helper that converts grid coordinates (i,j) back to the flat node index
    # all the nodes where i=0 are the left node of their cell, all nodes where i=nx are the right nodes of their cell,
    # all the nodes where j=0 are the bottom nodes of their cell, all the nodes where j=ny are the top nodes of their cell
    boundary = {
        "left":   [idx(0, j) for j in range(ny + 1)],
        "right":  [idx(nx, j) for j in range(ny + 1)],
        "bottom": [idx(i, 0) for i in range(nx + 1)],
        "top":    [idx(i, ny) for i in range(nx + 1)],
    }
    return FEMMesh(nodes, elements, boundary_nodes=boundary)  # wrap into the FEMMesh object

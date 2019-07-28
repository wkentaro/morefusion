import collections

import numpy as np
import trimesh


def matrix_from_points(points, origin, pitch, dims):
    indices = trimesh.voxel.points_to_indices(points, pitch, origin)

    if isinstance(dims, int):
        dims = (dims,) * 3
    if len(dims) != 3:
        raise TypeError('dims must be int or (3,) array-like')

    matrix = np.zeros(dims, dtype=bool)

    I = np.clip(indices[:, 0], 0, dims[0] - 1)
    J = np.clip(indices[:, 1], 0, dims[1] - 1)
    K = np.clip(indices[:, 2], 0, dims[2] - 1)

    matrix[I, J, K] = True

    return matrix

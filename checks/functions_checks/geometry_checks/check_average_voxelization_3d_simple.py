#!/usr/bin/env python

import numpy as np
import cupy

import objslampp

values = np.array([[1.], [1.25]], dtype=np.float32)
points = np.array([[1.5, 1.5, 1.5], [0.6, 0.6, 0.6]], dtype=np.float32)
matrix1, counts1 = objslampp.functions.average_voxelization_3d(
    values,
    points,
    origin=(0, 0, 0),
    pitch=1,
    dimensions=(3, 3, 3),
    return_counts=True,
)
values = cupy.array([[1.], [1.25]], dtype=np.float32)
points = cupy.array([[1.5, 1.5, 1.5], [0.6, 0.6, 0.6]], dtype=np.float32)
matrix2, counts2 = objslampp.functions.average_voxelization_3d(
    values,
    points,
    origin=(0, 0, 0),
    pitch=1,
    dimensions=(3, 3, 3),
    return_counts=True,
)

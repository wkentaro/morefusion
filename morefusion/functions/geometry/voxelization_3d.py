import chainer
import numpy as np


class Voxelization3D(chainer.Function):
    def __init__(self, *, batch_size, pitch, origin, dimensions):
        self.batch_size = batch_size
        self.pitch = pitch
        self.origin = origin

        if not (
            isinstance(dimensions, tuple)
            and len(dimensions) == 3
            and all(isinstance(d, int) for d in dimensions)
        ):
            raise ValueError("dimensions must be a tuple of 4 integers")

        self.dimensions = dimensions

    def check_type_forward(self, in_types):
        values_type, points_type, batch_indices_type = in_types[:3]
        chainer.utils.type_check.expect(
            values_type.dtype == np.float32,
            values_type.ndim == 2,
            points_type.dtype == np.float32,
            points_type.ndim == 2,
            points_type.shape[1] == 3,
            points_type.shape[0] == values_type.shape[0],
            batch_indices_type.dtype == np.int32,
            batch_indices_type.ndim == 1,
            batch_indices_type.shape[0] == values_type.shape[0],
        )

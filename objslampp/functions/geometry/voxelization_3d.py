import chainer
import numpy as np


class Voxelization3D(chainer.Function):

    def __init__(self, *, pitch, origin, dimensions, channels):
        self.pitch = pitch
        self.origin = origin

        if not (isinstance(dimensions, tuple) and
                len(dimensions) == 3 and
                all(isinstance(d, int) for d in dimensions)):
            raise ValueError('dimensions must be a tuple of 4 integers')

        if not isinstance(channels, int):
            raise ValueError('channels must be an integer')

        self.dimensions = dimensions
        self.channels = channels

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 2)

        values_type, points_type = in_types
        chainer.utils.type_check.expect(
            values_type.dtype == np.float32,
            values_type.ndim == 2,
            values_type.shape[0] == points_type.shape[0],
            values_type.shape[1] == self.channels,
            points_type.dtype == np.float32,
            points_type.ndim == 2,
            points_type.shape[1] == 3,
        )

import numbers

import numpy as np

from .base import SceneGenerationBase


class PlaneTypeSceneGeneration(SceneGenerationBase):
    def __init__(self, *args, **kwargs):
        extents = kwargs.pop("extents", (0.5, 0.5, 0.5))
        assert len(extents) == 3
        assert all(isinstance(x, numbers.Number) for x in extents)
        self._extents = np.asarray(extents, dtype=float)

        super().__init__(*args, **kwargs)

    def init_space(self):
        xlen, ylen, zlen = self._extents
        aabb_min = -xlen / 2, -ylen / 2, 0
        aabb_max = xlen / 2, ylen / 2, zlen
        self._aabb = aabb_min, aabb_max

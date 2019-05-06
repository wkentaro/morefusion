import numbers
import pathlib

import numpy as np
import pybullet
import trimesh

import objslampp

from .base import SceneGenerationBase


class BinTypeSceneGeneration(SceneGenerationBase):

    def __init__(self, *args, **kwargs):
        extents = kwargs.pop('extents', (0.5, 0.5, 0.2))
        assert len(extents) == 3
        assert all(isinstance(x, numbers.Number) for x in extents)
        self._extents = np.asarray(extents, dtype=float)

        thickness = kwargs.pop('thickness', 0.01)
        assert isinstance(thickness, numbers.Number)
        self._thickness = thickness

        super().__init__(*args, **kwargs)

    def init_space(self):
        cad = objslampp.extra.trimesh.bin_model(
            extents=self._extents,
            thickness=self._thickness,
        )

        cache_dir = pathlib.Path('/tmp')
        cad_file = cache_dir / f'{cad.md5()}.obj'
        if not cad_file.exists():
            trimesh.exchange.export.export_mesh(cad, str(cad_file))

        unique_id = objslampp.extra.pybullet.add_model(
            visual_file=cad_file,
            collision_file=objslampp.utils.get_collision_file(cad_file),
            position=(0, 0, self._extents[2] / 2),
        )
        self._simulate(nstep=100)

        aabb_min, aabb_max = pybullet.getAABB(unique_id)
        aabb_min, aabb_max = self._shrink_aabb(aabb_min, aabb_max, ratio=0.1)
        aabb_max[2] *= 1.5

        self._objects[unique_id] = dict(class_id=0)
        self._aabb = aabb_min, aabb_max

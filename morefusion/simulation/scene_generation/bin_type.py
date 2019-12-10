import numbers

import numpy as np
import path
import trimesh

import morefusion

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

        self._temp_dir = path.TempDir()

        super().__init__(*args, **kwargs)

    def __del__(self):
        self._temp_dir.rmtree_p()

    def init_space(self):
        import pybullet

        cad = morefusion.extra.trimesh.bin_model(
            extents=self._extents,
            thickness=self._thickness,
        )

        cad_file = self._temp_dir / f'{cad.md5()}.obj'
        if not cad_file.exists():
            trimesh.exchange.export.export_mesh(cad, str(cad_file))

        unique_id = morefusion.extra.pybullet.add_model(
            visual_file=cad_file,
            collision_file=morefusion.utils.get_collision_file(cad_file),
            position=(0, 0, self._extents[2] / 2),
            base_mass=10,
        )
        self._simulate(nstep=100)

        aabb_min, aabb_max = pybullet.getAABB(unique_id)
        aabb_min, aabb_max = self._shrink_aabb(aabb_min, aabb_max, ratio=0.1)
        aabb_max[2] *= 1.1

        self._objects[unique_id] = dict(class_id=0, cad_file=cad_file)
        self._aabb = aabb_min, aabb_max

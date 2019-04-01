import pathlib

import numpy as np
import pybullet
import trimesh

import objslampp

from .base import SceneGenerationBase
from ... import extra


class BinTypeSceneGeneration(SceneGenerationBase):

    def init_space(self):
        extents = np.array([0.5, 0.5, 0.2])
        cad = extra.trimesh.bin_model(extents=extents, thickness=0.01)

        cache_dir = pathlib.Path('/tmp')
        cad_file = cache_dir / f'{cad.md5()}.obj'
        if not cad_file.exists():
            trimesh.exchange.export.export_mesh(cad, str(cad_file))

        container_id = objslampp.extra.pybullet.add_model(
            visual_file=cad_file,
            collision_file=self._get_collision_file(cad_file),
            position=(0, 0, extents[2] / 2),
        )
        self._simulate(nstep=100)

        aabb_min, aabb_max = pybullet.getAABB(container_id)
        aabb_min, aabb_max = self._shrink_aabb(aabb_min, aabb_max, ratio=0.1)

        self._objects[container_id] = dict(class_id=0)
        self._aabb = aabb_min, aabb_max


if __name__ == '__main__':
    models = objslampp.datasets.YCBVideoModels()

    random_state = np.random.RandomState(0)
    generator = BinTypeSceneGeneration(
        models=models, n_object=10, random_state=random_state
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-45,
        cameraTargetPosition=(0, 0, 0),
    )
    generator.generate()

    eye = objslampp.geometry.points_from_angles(
        distance=[1], elevation=[45], azimuth=[45],
    )[0]
    T_camera2world = objslampp.geometry.look_at(
        eye=eye,
        at=(0, 0, 0),
        up=(0, -1, 0),
    )
    generator.debug_render(T_camera2world)

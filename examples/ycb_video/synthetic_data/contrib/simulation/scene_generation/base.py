import pathlib
import shlex
import subprocess

import imgviz
import numpy as np
import pybullet
import termcolor

import objslampp

from ... import geometry as geometry_module


class SceneGenerationBase:

    def __init__(self, models, n_object, *, random_state=None):
        self._models = models
        self._n_object = n_object
        if random_state is None:
            random_state = np.random.RandomState()
        self._random_state = random_state

        self._objects = {}
        self._aabb = (None, None)

        # launch simulator
        objslampp.extra.pybullet.init_world()

    @staticmethod
    def _get_collision_file(visual_file):
        visual_file = pathlib.Path(visual_file)
        name = visual_file.name
        name_noext, ext = name.rsplit('.')
        collision_file = visual_file.parent / (name_noext + '_convex.' + ext)
        if not collision_file.exists():
            cmd = f'testVHACD --input {visual_file} --output {collision_file}'\
                  ' --log /tmp/testVHACD.log --resolution 200000'
            # print(f'+ {cmd}')
            subprocess.check_output(shlex.split(cmd))
        return collision_file

    @staticmethod
    def _shrink_aabb(aabb_min, aabb_max, ratio):
        assert 0 <= ratio <= 1
        aabb_min = np.asarray(aabb_min)
        aabb_max = np.asarray(aabb_max)
        shrink_vec = aabb_max - aabb_min
        aabb_min = aabb_min + shrink_vec * ratio
        aabb_max = aabb_max - shrink_vec * ratio
        return aabb_min, aabb_max

    def _is_contained(self, unique_id):
        threshold = 0.1
        ratio = objslampp.extra.pybullet.aabb_contained_ratio(
            self._aabb, unique_id,
        )
        return ratio >= threshold

    def _simulate(self, nstep, fix=None):
        poses = {}
        if fix is not None:
            for unique_id in fix:
                pose = pybullet.getBasePositionAndOrientation(unique_id)
                poses[unique_id] = pose

        for _ in range(nstep):
            for unique_id, pose in poses.items():
                pybullet.resetBasePositionAndOrientation(
                    unique_id, *pose
                )
            pybullet.stepSimulation()

    def _is_colliding(self, unique_id):
        # check collision
        is_colliding = False
        for other_unique_id in objslampp.extra.pybullet.unique_ids:
            if other_unique_id == unique_id:
                continue
            points = pybullet.getClosestPoints(
                other_unique_id, unique_id, distance=0
            )
            distances = [pt[8] for pt in points]
            if any(d < 0 for d in distances):
                is_colliding = True
        return is_colliding

    def _spawn_object(self, class_id):
        termcolor.cprint(
            f'==> Spawning a new object: {class_id:04d}',
            attrs={'bold': True},
        )

        cad_file = self._models.get_cad_model(class_id=class_id)
        unique_id = objslampp.extra.pybullet.add_model(
            visual_file=cad_file,
            collision_file=self._get_collision_file(cad_file),
        )
        for _ in range(100):  # n_trial
            position = self._random_state.uniform(*self._aabb)
            orientation = self._random_state.uniform(-1, 1, (4,))
            pybullet.resetBasePositionAndOrientation(
                unique_id, position, orientation
            )

            if self._is_colliding(unique_id=unique_id):
                continue

            self._simulate(nstep=1000, fix=self._objects.keys())

            if not self._is_contained(unique_id=unique_id):
                continue

            self._objects[unique_id] = dict(class_id=class_id)
            break
        else:
            pybullet.removeBody(unique_id)

    def generate(self):
        termcolor.cprint(
            f'==> Started SceneGeneration: {self.__class__.__name__}',
            attrs={'bold': True},
        )
        self.init_space()

        class_ids = self._random_state.randint(
            1, self._models.n_class, self._n_object
        )
        termcolor.cprint(
            f'==> Selected Classes: {class_ids}', attrs={'bold': True}
        )

        for class_id in class_ids:
            self._spawn_object(class_id=class_id)
        termcolor.cprint('==> Finished scene generation', attrs={'bold': True})

    def render(self, T_camera2world, fovy, height, width):
        rgb, depth, ins = objslampp.extra.pybullet.render_camera(
            T_camera2world, fovy, height=height, width=width
        )
        cls = np.zeros_like(ins)
        for ins_id, data in self._objects.items():
            cls[ins == ins_id] = data['class_id']
        return rgb, depth, ins, cls

    def debug_render(self, T_camera2world):
        class_names = self._models.class_names

        height, width = 480, 640
        fovx = 60
        fovy = fovx / width * height

        scene = objslampp.extra.pybullet.get_trimesh_scene()
        list(scene.geometry.values())[0].visual.face_colors = (1., 1., 1.)
        for name, geometry in scene.geometry.items():
            if hasattr(geometry.visual, 'to_color'):
                geometry.visual = geometry.visual.to_color()
        scene.camera.resolution = (width, height)
        scene.camera.fov = (fovx, fovy)
        scene.camera.transform = objslampp.extra.trimesh.camera_transform(
            T_camera2world
        )

        rgb, depth, ins, cls = self.render(
            T_camera2world,
            fovy=scene.camera.fov[1],
            height=height,
            width=width,
        )

        ins_viz = imgviz.label2rgb(ins + 1, rgb)
        cls_viz = imgviz.label2rgb(
            cls, rgb, label_names=class_names, font_size=20
        )
        viz = imgviz.tile(
            [rgb, ins_viz, cls_viz], border=(255, 255, 255), shape=(1, 3)
        )
        viz = imgviz.resize(viz, width=1500)
        imgviz.io.pyglet_imshow(viz, 'pybullet')

        rgb = objslampp.extra.trimesh.save_image(scene)[:, :, :3]
        ins_viz = imgviz.label2rgb(ins + 1, rgb)
        cls_viz = imgviz.label2rgb(
            cls, rgb, label_names=class_names, font_size=20
        )
        viz = imgviz.tile(
            [rgb, ins_viz, cls_viz], border=(255, 255, 255), shape=(1, 3)
        )
        viz = imgviz.resize(viz, width=1500)
        imgviz.io.pyglet_imshow(viz, 'trimesh')

        imgviz.io.pyglet_run()

    def random_camera_trajectory(self, n_keypoints=8, n_points=64):
        # targets
        targets = self._random_state.uniform(*self._aabb, (n_keypoints, 3))
        targets = geometry_module.trajectory.sort(targets)
        targets = geometry_module.trajectory.interpolate(
            targets, n_points=n_points
        )

        # eyes
        distance = np.full((n_keypoints,), 1, dtype=float)
        elevation = self._random_state.uniform(30, 90, (n_keypoints,))
        azimuth = self._random_state.uniform(0, 360, (n_keypoints,))
        eyes = objslampp.geometry.points_from_angles(
            distance, elevation, azimuth
        )
        indices = np.linspace(0, n_points - 1, num=len(eyes))
        indices = indices.round().astype(int)
        eyes = geometry_module.trajectory.sort_by(eyes, key=targets[indices])
        eyes = geometry_module.trajectory.interpolate(eyes, n_points=n_points)

        Ts_cam2world = np.zeros((n_points, 4, 4), dtype=float)
        for i in range(n_points):
            Ts_cam2world[i] = objslampp.geometry.look_at(eyes[i], targets[i])
        return Ts_cam2world

    def init_space(self):
        raise NotImplementedError

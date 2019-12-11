import frozendict
import imgviz
import numpy as np
import termcolor
import trimesh

import morefusion


class SceneGenerationBase:

    def __init__(
        self,
        models,
        n_object,
        *,
        random_state=None,
        class_weight=None,
        multi_instance=True,
        connection_method=None,
        mesh_scale=None,
        n_trial=100,
    ):
        self._models = models
        self._n_object = n_object
        if random_state is None:
            random_state = np.random.mtrand._rand
        self._random_state = random_state
        self._class_weight = class_weight
        self._multi_instance = multi_instance
        if mesh_scale is not None:
            assert isinstance(mesh_scale, tuple)
            assert len(mesh_scale) == 2
            assert len(mesh_scale[0]) == 3
            assert len(mesh_scale[1]) == 3
        self._mesh_scale = mesh_scale
        self._n_trial = n_trial

        self._objects = {}
        self._aabb = (None, None)
        self._scene = None

        # launch simulator
        morefusion.extra.pybullet.init_world(
            connection_method=connection_method
        )

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
        threshold = 0.5
        ratio = morefusion.extra.pybullet.aabb_contained_ratio(
            self._aabb, unique_id,
        )
        return ratio >= threshold

    def _simulate(self, nstep, fix=None):
        import pybullet

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
        import pybullet

        # check collision
        is_colliding = False
        for other_unique_id in morefusion.extra.pybullet.unique_ids:
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
        import pybullet

        cad_ids = self._models.get_cad_ids(class_id=class_id)
        cad_id = self._random_state.choice(cad_ids, 1).item()
        cad_file = self._models.get_cad_file_from_id(cad_id=cad_id)

        termcolor.cprint(
            f'==> Spawning a new object: class_id={class_id:04d}, cad_id={cad_id}',  # NOQA
            attrs={'bold': True},
        )

        if self._mesh_scale is not None:
            mesh_scale = np.random.uniform(
                self._mesh_scale[0], self._mesh_scale[1]
            )
        unique_id = morefusion.extra.pybullet.add_model(
            visual_file=cad_file,
            collision_file=morefusion.utils.get_collision_file(cad_file),
            mesh_scale=mesh_scale
        )
        for _ in range(self._n_trial):
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

            self._objects[unique_id] = dict(
                class_id=class_id,
                cad_id=cad_id,
                mesh_scale=mesh_scale,
            )
            break
        else:
            pybullet.removeBody(unique_id)

    def generate(self):
        termcolor.cprint(
            f'==> Started SceneGeneration: {self.__class__.__name__}',
            attrs={'bold': True},
        )
        self.init_space()

        class_ids = self._random_state.choice(
            np.arange(1, self._models.n_class),
            self._n_object,
            replace=self._multi_instance,  # allow duplicates or not
            p=self._class_weight,
        )
        termcolor.cprint(
            f'==> Selected Classes: {class_ids}', attrs={'bold': True}
        )

        for class_id in class_ids:
            self._spawn_object(class_id=class_id)
        self._simulate(nstep=10000)
        termcolor.cprint('==> Finished scene generation', attrs={'bold': True})

    @property
    def unique_ids(self):
        return tuple(sorted(self._objects.keys()))

    def unique_id_to_cad_id(self, unique_id):
        if unique_id in self._objects:
            return self._objects[unique_id].get('cad_id', '')
        return ''

    def unique_ids_to_cad_ids(self, unique_ids):
        return np.array([self.unique_id_to_cad_id(u) for u in unique_ids])

    def unique_id_to_class_id(self, unique_id):
        if unique_id in self._objects:
            return self._objects[unique_id]['class_id']
        else:
            return 0  # background

    def unique_ids_to_class_ids(self, unique_ids):
        return np.array(
            [self.unique_id_to_class_id(i) for i in unique_ids],
            dtype=np.int32,
        )

    def unique_id_to_pose(self, unique_id):
        import pybullet

        pos, ori = pybullet.getBasePositionAndOrientation(unique_id)
        R_cad2world = pybullet.getMatrixFromQuaternion(ori)
        R_cad2world = np.asarray(R_cad2world, dtype=float).reshape(3, 3)
        t_cad2world = np.asarray(pos, dtype=float)
        T_cad2world = morefusion.geometry.compose_transform(
            R=R_cad2world, t=t_cad2world
        )
        return T_cad2world

    def unique_ids_to_poses(self, unique_ids):
        return np.array(
            [self.unique_id_to_pose(i) for i in unique_ids],
            dtype=float,
        )

    def unique_id_to_scale(self, unique_id):
        if unique_id in self._objects:
            return self._objects[unique_id].get('mesh_scale', (1, 1, 1))
        return (1, 1, 1)

    def unique_ids_to_scales(self, unique_ids):
        return np.array(
            [self.unique_id_to_scale(i) for i in unique_ids],
            dtype=float,
        )

    @property
    def scene(self):
        import pyrender

        if self._scene is not None:
            return self._scene

        scene = pyrender.Scene(bg_color=(0, 0, 0))
        for ins_id, data in self._objects.items():
            if 'cad_file' in data:
                cad_file = data['cad_file']
            else:
                assert data['class_id'] != 0
                cad_file = self._models.get_cad_file(class_id=data['class_id'])
            cad = trimesh.load_mesh(cad_file, process=False)
            try:
                obj = pyrender.Mesh.from_trimesh(cad, smooth=True)
            except ValueError:
                obj = pyrender.Mesh.from_trimesh(cad, smooth=False)
            pose = self.unique_id_to_pose(ins_id)
            scene.add(obj, pose=pose)

        self._scene = scene
        self._objects = frozendict.frozendict(self._objects)
        return self._scene

    def _render_pyrender(self, T_camera2world, fovy, height, width):
        # FIXME: pyrender and pybullet images are not perfectly aligned
        raise NotImplementedError

        import pyrender

        scene = self.scene
        node_camera = scene.add(
            obj=pyrender.PerspectiveCamera(
                yfov=np.deg2rad(fovy), aspectRatio=width / height
            ),
            pose=morefusion.extra.trimesh.to_opengl_transform(T_camera2world),
        )
        for _ in range(4):
            direction = self._random_state.uniform(-1, 1, (3,))
            direction /= np.linalg.norm(direction)
            scene.add(
                obj=pyrender.DirectionalLight(
                    intensity=self._random_state.uniform(0.5, 5),
                ),
                pose=trimesh.transformations.rotation_matrix(
                    angle=np.deg2rad(self._random_state.uniform(0, 45)),
                    direction=direction,
                ),
                parent_node=node_camera,
            )

        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, viewport_height=height
        )
        rgb, depth = renderer.render(scene)

        scene.remove_node(node_camera)
        return rgb, depth

    def _render_pybullet(self, T_camera2world, fovy, height, width):
        rgb, depth, ins = morefusion.extra.pybullet.render_camera(
            T_camera2world, fovy, height=height, width=width
        )
        cls = np.zeros_like(ins)
        for uid in self._objects:
            cls[ins == uid] = self.unique_id_to_class_id(unique_id=uid)
        return rgb, depth, ins, cls

    def render(self, *args, **kwargs):
        rgb, depth, ins, cls = self._render_pybullet(*args, **kwargs)

        return rgb, depth, ins, cls

    def debug_render(self, T_camera2world):
        class_names = self._models.class_names

        height, width = 480, 640
        fovx = 60
        fovy = fovx / width * height

        scene = morefusion.extra.pybullet.get_trimesh_scene()
        list(scene.geometry.values())[0].visual.face_colors = (1., 1., 1.)
        for name, geometry in scene.geometry.items():
            if hasattr(geometry.visual, 'to_color'):
                geometry.visual = geometry.visual.to_color()
        scene.camera.resolution = (width, height)
        scene.camera.fov = (fovx, fovy)
        scene.camera_transform = morefusion.extra.trimesh.to_opengl_transform(
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

        rgb = morefusion.extra.trimesh.save_image(scene)[:, :, :3]
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

    def get_aabb(self):
        import pybullet

        aabb_min = None
        aabb_max = None
        for unique_id in self._objects:
            aabb = pybullet.getAABB(unique_id)
            if aabb_min is None:
                aabb_min = aabb[0]
            else:
                aabb_min = np.minimum(aabb_min, aabb[0])
            if aabb_max is None:
                aabb_max = aabb[1]
            else:
                aabb_max = np.maximum(aabb_max, aabb[1])
        return tuple(aabb_min), tuple(aabb_max)

    def random_camera_trajectory(
        self, n_keypoints=8, n_points=64, distance=(1, 1), elevation=(45, 90)
    ):
        aabb = self.get_aabb()

        # targets
        targets = self._random_state.uniform(*aabb, (n_keypoints, 3))
        targets = morefusion.geometry.trajectory.sort(targets)
        targets = morefusion.geometry.trajectory.interpolate(
            targets, n_points=n_points
        )

        # eyes
        distance = self._random_state.uniform(
            distance[0], distance[1], n_keypoints
        )
        elevation = self._random_state.uniform(
            elevation[0], elevation[1], n_keypoints
        )
        azimuth = self._random_state.uniform(0, 360, n_keypoints)
        eyes = morefusion.geometry.points_from_angles(
            distance, elevation, azimuth
        )
        indices = np.linspace(0, n_points - 1, num=len(eyes))
        indices = indices.round().astype(int)
        eyes = morefusion.geometry.trajectory.sort_by(
            eyes, key=targets[indices]
        )
        eyes = morefusion.geometry.trajectory.interpolate(
            eyes, n_points=n_points
        )

        Ts_cam2world = np.zeros((n_points, 4, 4), dtype=float)
        for i in range(n_points):
            Ts_cam2world[i] = morefusion.geometry.look_at(eyes[i], targets[i])
        return Ts_cam2world

    def init_space(self):
        raise NotImplementedError

#!/usr/bin/env python

import chainer
from chainer.backends import cuda
import chainer.functions as F
import glooey
import imgviz
import numpy as np
import pyglet
import trimesh
import trimesh.transformations as tf

import objslampp

import contrib


class OccupancyGridAlignmentModel(chainer.Link):

    def __init__(self, quaternion_init=None, translation_init=None):
        super().__init__()
        with self.init_scope():
            if quaternion_init is None:
                quaternion_init = np.array([1, 0, 0, 0], dtype=np.float32)
            self.quaternion = chainer.Parameter(
                initializer=quaternion_init
            )
            if translation_init is None:
                translation_init = np.zeros((3,), dtype=np.float32)
            self.translation = chainer.Parameter(
                initializer=translation_init
            )

    def forward(
        self,
        points_source,
        grid_target,
        *,
        pitch,
        origin,
        threshold,
    ):
        transform = objslampp.functions.quaternion_matrix(
            self.quaternion[None]
        )
        transform = objslampp.functions.compose_transform(
            transform[:, :3, :3], self.translation[None]
        )

        points_source = objslampp.functions.transform_points(
            points_source, transform
        )[0]
        grid_source = objslampp.functions.occupancy_grid_3d(
            points_source,
            pitch=pitch,
            origin=origin,
            dimension=grid_target.shape[1:],
            threshold=threshold,
        )

        assert grid_target.dtype == np.float32
        occupied_target = grid_target[0]
        intersection = F.sum(occupied_target * grid_source)
        denominator = F.sum(occupied_target)
        reward = intersection / denominator

        assert grid_target.dtype == np.float32
        unoccupied_target = np.maximum(grid_target[1], grid_target[2])
        unoccupied_target = unoccupied_target.astype(np.float32)
        intersection = F.sum(unoccupied_target * grid_source)
        denominator = F.sum(grid_source)
        penalty = intersection / denominator

        loss = - reward + penalty
        return loss


class InstanceOccupancyGridRegistration:

    def __init__(
        self,
        points_source,
        grid_target,
        *,
        pitch,
        origin,
        threshold,
        transform_init,
        gpu=0,
    ):
        quaternion_init = tf.quaternion_from_matrix(transform_init)
        quaternion_init = quaternion_init.astype(np.float32)
        translation_init = tf.translation_from_matrix(transform_init)
        translation_init = translation_init.astype(np.float32)

        model = OccupancyGridAlignmentModel(quaternion_init, translation_init)

        self._grid_target_cpu = grid_target

        if gpu >= 0:
            model.to_gpu(gpu)
            points_source = model.xp.asarray(points_source)
            grid_target = model.xp.asarray(grid_target)

        self._points_source = points_source
        self._grid_target = grid_target
        self._pitch = pitch
        self._origin = origin
        self._threshold = threshold

        self._optimizer = chainer.optimizers.Adam(alpha=0.1)
        self._optimizer.setup(model)
        model.translation.update_rule.hyperparam.alpha *= 0.1

        self._iteration = -1

    def step(self):
        self._iteration += 1

        model = self._optimizer.target

        loss = model(
            points_source=self._points_source,
            grid_target=self._grid_target,
            pitch=self._pitch,
            origin=self._origin,
            threshold=self._threshold,
        )
        loss.backward()
        self._optimizer.update()
        model.cleargrads()

        loss = float(loss.array)

        # print(f'[{self._iteration:08d}] {loss}')
        # print(f'quaternion:', model.quaternion.array.tolist())
        # print(f'translation:', model.translation.array.tolist())

    @property
    def transform(self):
        model = self._optimizer.target
        quaternion = cuda.to_cpu(model.quaternion.array)
        translation = cuda.to_cpu(model.translation.array)
        transform = tf.quaternion_matrix(quaternion)
        transform = objslampp.geometry.compose_transform(
            transform[:3, :3], translation
        )
        return transform

    def nstep(self, iteration):
        yield self.transform
        for _ in range(iteration):
            self.step()
            yield self.transform

    def visualize(self, cad, T_cad2cam_true, T_cad2cam_pred):
        scenes = {}

        grid_target = self._grid_target_cpu
        pitch = self._pitch
        origin = self._origin

        scene = trimesh.Scene()
        # occupied target/untarget
        voxel = trimesh.voxel.Voxel(
            matrix=grid_target[0], pitch=pitch, origin=origin
        )
        geom = voxel.as_boxes((1., 0, 0, 0.5))
        scene.add_geometry(geom, geom_name='occupied_target')
        voxel = trimesh.voxel.Voxel(
            matrix=grid_target[1],
            pitch=pitch,
            origin=origin,
        )
        geom = voxel.as_boxes((0, 1., 0, 0.5))
        scene.add_geometry(geom, geom_name='occupied_untarget')
        scenes['instance_occupied'] = scene

        # empty
        scene = trimesh.Scene()
        voxel = trimesh.voxel.Voxel(
            matrix=grid_target[2], pitch=pitch, origin=origin
        )
        geom = voxel.as_boxes((0.5, 0.5, 0.5, 0.5))
        scene.add_geometry(geom, geom_name='empty')
        scenes['instance_empty'] = scene

        scene = trimesh.Scene()
        # cad_true
        cad_trans = cad.copy()
        cad_trans.visual.vertex_colors[:, 3] = 127
        scene.add_geometry(
            cad_trans,
            transform=T_cad2cam_true,
            geom_name='cad_true',
            node_name='cad_true',
        )
        scenes['instance_cad'] = scene

        # cad_pred
        for scene in scenes.values():
            scene.add_geometry(
                cad,
                transform=T_cad2cam_pred,
                geom_name='cad_pred',
                node_name='cad_pred',
            )

        # bbox
        aabb_min = origin - pitch / 2
        aabb_max = aabb_min + pitch * np.array(grid_target.shape[1:])
        geom = trimesh.path.creation.box_outline(aabb_max - aabb_min)
        geom.apply_translation((aabb_min + aabb_max) / 2)
        for scene in scenes.values():
            scene.add_geometry(geom, geom_name='bbox')

        return scenes


class OccupancyGridRegistration:

    _models = objslampp.datasets.YCBVideoModels()

    def __init__(
        self,
        rgb,
        pcd,
        instance_label,
        instance_ids,
        class_ids,
        Ts_cad2cam_true,
        Ts_cad2cam_pred=None,
    ):
        N = len(instance_ids)
        assert instance_ids.shape == (N,) and 0 not in instance_ids
        assert class_ids.shape == (N,) and 0 not in class_ids
        assert Ts_cad2cam_true.shape == (N, 4, 4)
        H, W = pcd.shape[:2]
        assert pcd.shape == (H, W, 3)
        assert instance_label.shape == (H, W)

        self._instance_ids = instance_ids
        self._class_ids = dict(zip(instance_ids, class_ids))
        self._Ts_cad2cam_true = dict(zip(instance_ids, Ts_cad2cam_true))
        self._rgb = rgb
        self._pcd = pcd
        self._instance_label = instance_label
        self._mapping = contrib.MultiInstanceOctreeMapping()

        pitch = 0.01
        nonnan = ~np.isnan(pcd).any(axis=2)
        for ins_id in np.unique(instance_label):
            if ins_id == -1:
                continue
            mask = instance_label == ins_id
            self._mapping.initialize(ins_id, pitch=pitch)
            self._mapping.integrate(ins_id, mask, pcd)

        self._cads = {}
        for instance_id in self._instance_ids:
            class_id = self._class_ids[instance_id]
            cad_file = self._models.get_cad_model(class_id=class_id)
            cad = trimesh.load(str(cad_file))
            cad.visual = cad.visual.to_color()
            self._cads[instance_id] = cad

        if Ts_cad2cam_pred is None:
            self._Ts_cad2cam_pred = {}
            nonnan = ~np.isnan(pcd).any(axis=2)
            for instance_id in instance_ids:
                mask = instance_label == instance_id
                centroid = pcd[nonnan & mask].mean(axis=0)
                T_cad2cam_pred = tf.translation_matrix(centroid)
                self._Ts_cad2cam_pred[instance_id] = T_cad2cam_pred
        else:
            assert len(instance_ids) == len(Ts_cad2cam_pred)
            self._Ts_cad2cam_pred = dict(zip(instance_ids, Ts_cad2cam_pred))

    def update_octree(self, instance_id):
        T_cad2cam_pred = self._Ts_cad2cam_pred[instance_id]

        class_id = self._class_ids[instance_id]
        pcd_file = self._models.get_pcd_model(class_id=class_id)
        points = np.loadtxt(pcd_file)
        points = tf.transform_points(points, T_cad2cam_pred)

        self._mapping.update(instance_id, points)

    def register_instance(self, instance_id):
        models = self._models

        # parameters
        threshold = 2
        dim = 20

        # scene-level data
        class_ids = self._class_ids
        pcd = self._pcd
        instance_label = self._instance_label

        # instance-level data
        class_id = class_ids[instance_id]
        cad_file = models.get_cad_model(class_id=class_id)
        diagonal = models.get_bbox_diagonal(cad_file)
        pitch = diagonal * 1.1 / dim
        mask = instance_label == instance_id
        centroid = np.nanmean(pcd[mask], axis=0)
        origin = centroid - ((dim / 2 - 0.5) * pitch)
        grid_target, grid_nontarget, grid_empty = \
            self._mapping.get_target_grids(
                target_id=instance_id,
                dimensions=(dim, dim, dim),
                pitch=pitch,
                origin=origin,
            )

        #
        pcd_file = models.get_pcd_model(class_id=class_id)
        points_source = np.loadtxt(pcd_file, dtype=np.float32)
        points_source = objslampp.extra.open3d.voxel_down_sample(
            points_source, voxel_size=pitch
        )
        points_source = points_source.astype(np.float32)

        self._instance_id = instance_id

        grids = np.stack((grid_target, grid_nontarget, grid_empty))
        grids = grids.astype(np.float32)
        registration_ins = InstanceOccupancyGridRegistration(
            points_source,
            grids,
            pitch=pitch,
            origin=origin,
            threshold=threshold,
            transform_init=self._Ts_cad2cam_pred[instance_id],
        )
        self._registration_ins = registration_ins

        Ts_cad2cam_pred = registration_ins.nstep(100)
        self._Ts_cad2cam_pred[instance_id] = next(Ts_cad2cam_pred)
        yield
        for T_cad2cam_pred in Ts_cad2cam_pred:
            self._Ts_cad2cam_pred[instance_id] = T_cad2cam_pred
            yield

    def visualize(self):  # NOQA
        # scene-level
        rgb = self._rgb
        pcd = self._pcd

        scenes = {}
        # scene_pcd
        scenes['scene_cad'] = trimesh.Scene()
        scenes['scene_pcd'] = trimesh.Scene()
        nonnan = ~np.isnan(pcd).any(axis=2)
        geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
        scenes['scene_pcd'].add_geometry(geom, geom_name='pcd')
        for instance_id in self._instance_ids:
            if instance_id not in self._cads:
                continue
            cad = self._cads[instance_id]
            T_cad2cam_pred = self._Ts_cad2cam_pred[instance_id]
            if cad:
                for key in ['scene_cad', 'scene_pcd']:
                    scenes[key].add_geometry(
                        cad,
                        transform=T_cad2cam_pred,
                        geom_name=f'cad_pred_{instance_id}',
                        node_name=f'cad_pred_{instance_id}',
                    )
        # scene_occupancy
        colormap = imgviz.label_colormap()
        scenes['scene_occupied'] = trimesh.Scene()
        scenes['scene_empty'] = trimesh.Scene()
        for instance_id in self._mapping.instance_ids:
            occupied, empty = self._mapping.get_target_pcds(instance_id)
            color = trimesh.visual.to_rgba(colormap[instance_id])
            color[3] = 127
            geom = trimesh.voxel.multibox(
                occupied, pitch=0.01, colors=color
            )
            scenes['scene_occupied'].add_geometry(
                geom, geom_name=f'occupied_{instance_id}'
            )
            geom = trimesh.PointCloud(vertices=empty, colors=[0.5, 0.5, 0.5])
            scenes['scene_empty'].add_geometry(
                geom, geom_name=f'empty_{instance_id}'
            )

        # instance-level
        instance_id = self._instance_id
        registration_ins = self._registration_ins
        cad = self._cads[instance_id]
        T_cad2cam_true = self._Ts_cad2cam_true[instance_id]
        T_cad2cam_pred = self._Ts_cad2cam_pred[instance_id]
        all_scenes = registration_ins.visualize(
            cad=cad,
            T_cad2cam_true=T_cad2cam_true,
            T_cad2cam_pred=T_cad2cam_pred,
        )
        all_scenes.update(scenes)

        # set camera
        camera = trimesh.scene.Camera(
            resolution=(640, 480),
            fov=(60 * 0.7, 45 * 0.7),
            transform=objslampp.extra.trimesh.camera_transform(),
        )
        for scene in all_scenes.values():
            scene.camera = camera
        return all_scenes


def refinement(
    instance_ids,
    class_ids,
    rgb,
    pcd,
    instance_label,
    Ts_cad2cam_true,
    Ts_cad2cam_pred=None,
    points_occupied=None,
):
    registration = OccupancyGridRegistration(
        rgb=rgb,
        pcd=pcd,
        instance_label=instance_label,
        instance_ids=instance_ids,
        class_ids=class_ids,
        Ts_cad2cam_true=Ts_cad2cam_true,
        Ts_cad2cam_pred=Ts_cad2cam_pred,
    )

    if points_occupied:
        for ins_id, points in points_occupied.items():
            registration._mapping.update(ins_id, points)

    coms = np.array([
        np.nanmedian(pcd[instance_label == i], axis=0) for i in instance_ids
    ])
    instance_ids = np.array(instance_ids)[np.argsort(coms[:, 2])]
    instance_ids = iter(instance_ids)

    # -------------------------------------------------------------------------

    nrow, ncol = 2, 4
    height = int(round(0.7 * 480)) * nrow
    width = int(round(0.7 * 640)) * ncol
    window = pyglet.window.Window(width=width, height=height)
    window.play = False
    window.result = registration.register_instance(next(instance_ids))
    next(window.result)
    window.rotate = 0

    print('''\
==> Usage
q: close window
n: next iteration
s: play iterations
z: reset camera
r/R: rotate camera
N: next instance''')

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()
            elif symbol == pyglet.window.key.S:
                window.play = not window.play
                print(f'==> window.play: {window.play}')
            elif symbol == pyglet.window.key.N:
                try:
                    next(window.result)
                except StopIteration:
                    on_key_press(
                        symbol=pyglet.window.key.N,
                        modifiers=pyglet.window.key.MOD_SHIFT,
                    )
                    return
            elif symbol == pyglet.window.key.Z:
                for widget in widgets.values():
                    camera = widget.scene.camera
                    camera.transform = \
                        objslampp.extra.trimesh.camera_transform()
        if symbol == pyglet.window.key.R:
            # rotate camera
            window.rotate = not window.rotate  # 0/1
            if modifiers == pyglet.window.key.MOD_SHIFT:
                window.rotate *= -1
        if modifiers == pyglet.window.key.MOD_SHIFT:
            if symbol == pyglet.window.key.N:
                try:
                    print('==> updating octrees')
                    registration.update_octree(registration._instance_id)
                    print('==> updated octrees')

                    instance_id = next(instance_ids)
                    print(f'==> initializing next instance: {instance_id}')
                    window.result = registration.register_instance(instance_id)
                    next(window.result)
                    print(f'==> initialized instance: {instance_id}')
                except StopIteration:
                    return

    def callback(dt, widgets=None):
        if window.rotate:
            point = np.mean([
                tf.translation_from_matrix(T)
                for T in registration._Ts_cad2cam_true.values()
            ], axis=0)
            for widget in widgets.values():
                camera = widget.scene.camera
                axis = tf.transform_points(
                    [[0, 1, 0]], camera.transform, translate=False
                )[0]
                camera.transform = tf.rotation_matrix(
                    np.deg2rad(window.rotate), axis, point=point
                ) @ camera.transform
            return
        if window.play:
            try:
                next(window.result)
            except StopIteration:
                on_key_press(
                    symbol=pyglet.window.key.N,
                    modifiers=pyglet.window.key.MOD_SHIFT,
                )
                return
        scenes = registration.visualize()
        if widgets:
            for key, widget in widgets.items():
                widget.scene.geometry.update(scenes[key].geometry)
                widget.scene.graph.load(scenes[key].graph.to_edgelist())
                widget._draw()
        return scenes

    gui = glooey.Gui(window)

    grid = glooey.Grid(num_rows=nrow, num_cols=ncol)
    grid.set_padding(5)
    vbox = glooey.VBox()
    vbox.add(glooey.Label('rgb', color=(255, 255, 255)), size=0)
    vbox.add(
        glooey.Image(
            objslampp.extra.pyglet.numpy_to_image(rgb), responsive=True
        )
    )
    grid.add(0, 0, vbox)
    widgets = {}
    scenes = callback(-1)
    for i, (key, scene) in enumerate(scenes.items()):
        i += 1
        widgets[key] = trimesh.viewer.SceneWidget(scene)
        vbox = glooey.VBox()
        vbox.add(glooey.Label(key, color=(255, 255, 255)), size=0)
        vbox.add(widgets[key])
        grid[i // ncol, i % ncol] = vbox
    gui.add(grid)

    pyglet.clock.schedule_interval(callback, 1 / 30, widgets)
    pyglet.app.run()
    pyglet.clock.unschedule(callback)

    return registration


def main():
    dataset = objslampp.datasets.YCBVideoDataset('train')
    frame = dataset.get_example(1000)

    # scene-level data
    instance_ids = class_ids = frame['meta']['cls_indexes']
    Ts_cad2cam_true = np.tile(np.eye(4), (len(instance_ids), 1, 1))
    Ts_cad2cam_true[:, :3, :4] = frame['meta']['poses'].transpose(2, 0, 1)
    K = frame['meta']['intrinsic_matrix']
    rgb = frame['color']
    pcd = objslampp.geometry.pointcloud_from_depth(
        frame['depth'], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    instance_label = frame['label']

    refinement(
        instance_ids=instance_ids,
        class_ids=class_ids,
        rgb=rgb,
        pcd=pcd,
        instance_label=instance_label,
        Ts_cad2cam_true=Ts_cad2cam_true,
    )


if __name__ == '__main__':
    main()

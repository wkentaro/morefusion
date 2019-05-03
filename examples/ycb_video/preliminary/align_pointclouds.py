#!/usr/bin/env python

import glooey
import imgviz
import numpy as np
import open3d
import pyglet
import trimesh
import trimesh.transformations as tf

import objslampp


class PointCloudRegistration:

    def __init__(
        self,
        rgb,
        pcd,
        instance_label,
        instance_ids,
        class_ids,
        Ts_cad2cam_true,
    ):
        self._rgb = rgb
        self._pcd = pcd
        self._instance_label = instance_label
        self._instance_label_viz = imgviz.label2rgb(instance_label)
        self._instance_ids = instance_ids
        self._class_ids = class_ids
        self._Ts_cad2cam_true = Ts_cad2cam_true

        self._cads = {}
        self._Ts_cam2cad_pred = {}

    def register_instance(self, instance_id):
        pcd = self._pcd
        nonnan = ~np.isnan(pcd).any(axis=2)

        models = objslampp.datasets.YCBVideoModels()

        index = np.where(self._instance_ids == instance_id)[0][0]
        class_id = self._class_ids[index]
        mask = self._instance_label == instance_id

        pcd_file = models.get_pcd_model(class_id=class_id)
        points_cad = np.loadtxt(pcd_file, dtype=np.float32)

        cad_file = models.get_cad_model(class_id=class_id)
        cad = trimesh.load(str(cad_file))
        cad.visual = cad.visual.to_color()

        target = open3d.PointCloud()
        target.points = open3d.Vector3dVector(points_cad)

        pcd_ins = pcd[nonnan & mask]
        centroid = pcd_ins.mean(axis=0)
        T_cam2com = tf.translation_matrix(- centroid)

        source = open3d.PointCloud()
        source.points = open3d.Vector3dVector(pcd_ins)

        source = open3d.voxel_down_sample(source, voxel_size=0.01)
        target = open3d.voxel_down_sample(target, voxel_size=0.01)

        self._source = source
        self._target = target
        self._instance_id = instance_id

        self._cads[instance_id] = cad
        self._Ts_cam2cad_pred[instance_id] = T_cam2com
        yield
        for i in range(300):
            result = open3d.registration_icp(
                source,  # points_from_depth
                target,  # points_from_cad
                0.02,
                self._Ts_cam2cad_pred[instance_id],
                open3d.TransformationEstimationPointToPoint(False),
                open3d.ICPConvergenceCriteria(max_iteration=1),
            )
            self._Ts_cam2cad_pred[instance_id] = result.transformation
            yield

    @property
    def T_cad2cam_pred(self):
        return np.linalg.inv(self._T_cam2cad_pred)

    def visualize(self):
        rgb = self._rgb
        pcd = self._pcd

        instance_id = self._instance_id
        index = np.where(self._instance_ids == instance_id)[0][0]
        T_cad2cam_true = self._Ts_cad2cam_true[index]

        cad = self._cads[instance_id]
        T_cad2cam_pred = np.linalg.inv(self._Ts_cam2cad_pred[instance_id])

        source = self._source
        target = self._target

        camera = trimesh.scene.Camera(
            resolution=(640, 480),
            fov=(60 * 0.7, 45 * 0.7),
            transform=objslampp.extra.trimesh.camera_transform(),
        )

        scenes = {}

        scenes['instance_points'] = trimesh.Scene(camera=camera)
        # points_source: points_from_depth
        geom = trimesh.PointCloud(
            vertices=np.asarray(source.points), colors=(0, 1., 0)
        )
        scenes['instance_points'].add_geometry(geom, geom_name='points_source')
        # points_target: points_from_cad
        geom = trimesh.PointCloud(
            vertices=np.asarray(target.points), colors=(1., 0, 0)
        )
        scenes['instance_points'].add_geometry(
            geom,
            transform=T_cad2cam_pred,
            geom_name='points_target',
            node_name='points_target',
        )

        scenes['instance_cad'] = trimesh.Scene(camera=camera)
        # cad_pred
        scenes['instance_cad'].add_geometry(
            cad,
            transform=T_cad2cam_pred,
            geom_name='cad_pred',
            node_name='cad_pred',
        )
        # cad_true
        cad_copy = cad.copy()
        cad_copy.visual.vertex_colors[:, 3] = 127
        scenes['instance_cad'].add_geometry(
            cad_copy,
            transform=T_cad2cam_true,
            geom_name='cad_true',
            node_name='cad_true',
        )

        # scene-level
        scenes['scene_pcd'] = trimesh.Scene(camera=camera)
        nonnan = ~np.isnan(pcd).any(axis=2)
        geom = trimesh.PointCloud(pcd[nonnan], colors=rgb[nonnan])
        scenes['scene_pcd'].add_geometry(geom, geom_name='pcd')
        for instance_id in self._instance_ids:
            if instance_id not in self._Ts_cam2cad_pred:
                continue
            assert instance_id in self._cads
            T_cad2cam_pred = np.linalg.inv(self._Ts_cam2cad_pred[instance_id])
            scenes['scene_pcd'].add_geometry(
                self._cads[instance_id],
                transform=T_cad2cam_pred,
                node_name=f'cad_pred_{instance_id}',
                geom_name=f'cad_pred_{instance_id}',
            )

        scenes['scene_label'] = trimesh.Scene(camera=camera)
        geom = trimesh.PointCloud(
            pcd[nonnan],
            colors=self._instance_label_viz[nonnan],
        )
        scenes['scene_label'].add_geometry(geom, geom_name='pcd')

        return scenes


def main():
    dataset = objslampp.datasets.YCBVideoDataset('train')

    frame = dataset[1000]

    class_ids = frame['meta']['cls_indexes']
    instance_ids = class_ids
    depth = frame['depth']
    K = frame['meta']['intrinsic_matrix']
    rgb = frame['color']
    pcd = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    instance_label = frame['label']
    Ts_cad2cam_true = np.tile(np.eye(4), (len(instance_ids), 1, 1))
    Ts_cad2cam_true[:, :3, :4] = frame['meta']['poses'].transpose(2, 0, 1)

    registration = PointCloudRegistration(
        rgb=rgb,
        pcd=pcd,
        instance_label=instance_label,
        instance_ids=instance_ids,
        class_ids=class_ids,
        Ts_cad2cam_true=Ts_cad2cam_true,
    )

    instance_ids = iter(instance_ids)

    # -------------------------------------------------------------------------

    def callback(dt, widget=None):
        if window.rotate:
            point = registration._Ts_cad2cam_true[:, :3, 3].mean(axis=0)
            for widget in widgets.values():
                camera = widget.scene.camera
                camera.transform = tf.rotation_matrix(
                    np.deg2rad(window.rotate), [0, 1, 0], point=point
                ) @ camera.transform
            return

        if window.play:
            try:
                next(window.result)
            except StopIteration:
                pyglet.clock.unschedule(callback)
                return

        scenes = registration.visualize()
        if widgets is not None:
            for key, widget in widgets.items():
                widget.scene.geometry.update(scenes[key].geometry)
                widget.scene.graph.load(scenes[key].graph.to_edgelist())
                widget._draw()

    nrow, ncol = 2, 2
    width = int(round(640 * 0.8 * ncol))
    height = int(round(480 * 0.8 * nrow))
    window = pyglet.window.Window(width=width, height=height)
    window.play = False
    window.rotate = 0

    window.result = registration.register_instance(next(instance_ids))
    next(window.result)
    scenes = registration.visualize()

    print('''\
==> Usage
q: close window
n: next iteration
s: play iterations
r: rotate camera
z: reset view
N: next instance''')

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()
            elif symbol == pyglet.window.key.N:
                next(window.result)
            elif symbol == pyglet.window.key.S:
                window.play = not window.play
            elif symbol == pyglet.window.key.Z:
                for widget in widgets.values():
                    camera = widget.scene.camera
                    camera.transform = \
                        objslampp.extra.trimesh.camera_transform()
        if symbol == pyglet.window.key.R:
            window.rotate = not window.rotate
            if modifiers == pyglet.window.key.MOD_SHIFT:
                window.rotate *= -1
        if modifiers == pyglet.window.key.MOD_SHIFT:
            if symbol == pyglet.window.key.N:
                instance_id = next(instance_ids)
                print(f'==> initializing next instance: {instance_id}')
                window.result = registration.register_instance(instance_id)
                next(window.result)
                print(f'==> initialized: {instance_id}')

    gui = glooey.Gui(window)
    grid = glooey.Grid(nrow, ncol)
    grid.set_padding(5)

    widgets = {}
    for i, (key, scene) in enumerate(scenes.items()):
        vbox = glooey.VBox()
        vbox.add(glooey.Label(key, color=(255, 255, 255)), size=0)
        widgets[key] = trimesh.viewer.SceneWidget(scene)
        vbox.add(widgets[key])
        grid.add(i // ncol, i % ncol, vbox)
    gui.add(grid)

    pyglet.clock.schedule_interval(callback, 1 / 30, widgets)
    pyglet.app.run()


if __name__ == '__main__':
    main()

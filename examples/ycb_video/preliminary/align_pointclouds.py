#!/usr/bin/env python

import glooey
import numpy as np
import pyglet
import trimesh
import trimesh.transformations as tf

import objslampp

import open3d


def main():
    models = objslampp.datasets.YCBVideoModels()
    dataset = objslampp.datasets.YCBVideoDataset('train')
    frame = dataset[1000]
    class_ids = frame['meta']['cls_indexes']
    instance_ids = class_ids
    depth = frame['depth']
    K = frame['meta']['intrinsic_matrix']
    pcd = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    nonnan = ~np.isnan(depth)
    instance_label = frame['label']

    index = 1
    instance_id = instance_ids[index]
    class_id = class_ids[index]
    T_cad2cam = np.r_[frame['meta']['poses'][:, :, index], [[0, 0, 0, 1]]]
    mask = instance_label == instance_id

    pcd_file = models.get_pcd_model(class_id=class_id)
    points_cad = np.loadtxt(pcd_file, dtype=np.float32)

    cad_file = models.get_cad_model(class_id=class_id)
    cad = trimesh.load(str(cad_file))
    cad.visual = cad.visual.to_color()

    target = open3d.PointCloud()
    target.points = open3d.Vector3dVector(points_cad)

    pcd_ins = pcd[nonnan & mask]
    centroid = pcd_ins.mean(axis=0)
    T_cam2centroid = tf.translation_matrix(- centroid)

    source = open3d.PointCloud()
    source.points = open3d.Vector3dVector(pcd_ins)

    source = open3d.voxel_down_sample(source, voxel_size=0.01)
    target = open3d.voxel_down_sample(target, voxel_size=0.01)

    def nstep(iteration):
        T_cam2cad_pred = T_cam2centroid
        yield T_cam2cad_pred
        for i in range(iteration):
            result = open3d.registration_icp(
                source,  # points_from_depth
                target,  # points_from_cad
                0.02,
                T_cam2cad_pred,
                open3d.TransformationEstimationPointToPoint(False),
                open3d.ICPConvergenceCriteria(max_iteration=1),
            )
            print(f'[{i:08d}]')
            T_cam2cad_pred = result.transformation
            print(T_cam2cad_pred)
            yield T_cam2cad_pred

    transforms = nstep(300)

    def callback(dt, widget=None):
        if widget and not window.play:
            return

        try:
            T_cam2cad_pred = next(transforms)
            T_cad2cam_pred = np.linalg.inv(T_cam2cad_pred)
        except StopIteration:
            pyglet.clock.unschedule(callback)
            return

        scenes = {}

        scene = trimesh.Scene()
        # points_source: points_from_depth
        geom = trimesh.PointCloud(
            vertices=np.asarray(source.points), colors=(0, 1., 0)
        )
        scene.add_geometry(geom, geom_name='points_source')
        # points_target: points_from_cad
        geom = trimesh.PointCloud(
            vertices=np.asarray(target.points), colors=(1., 0, 0)
        )
        geom.apply_transform(T_cad2cam_pred)
        scene.add_geometry(geom, geom_name='points_target')
        scenes['points'] = scene

        scene = trimesh.Scene()
        # cad_pred
        cad_copy = cad.copy()
        cad_copy.apply_transform(T_cad2cam_pred)
        scene.add_geometry(cad_copy, geom_name='cad_pred')
        # cad_true
        cad_copy = cad.copy()
        cad_copy.visual.vertex_colors[:, 3] = 127
        cad_copy.apply_transform(T_cad2cam)
        scene.add_geometry(cad_copy, geom_name='cad_true')
        scenes['cad'] = scene

        center = scenes['cad'].centroid
        for scene in scenes.values():
            scene.set_camera(
                angles=[np.deg2rad(180), 0, 0],
                distance=0.6,
                center=center,
            )

        if widgets is not None:
            for key, widget in widgets.items():
                widget.scene.geometry.update(scenes[key].geometry)
                widget.scene.graph.load(scenes[key].graph.to_edgelist())
                widget._draw()
        return scenes

    window = pyglet.window.Window(width=640 * 2, height=480)
    window.play = False

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()
            elif symbol == pyglet.window.key.S:
                window.play = not window.play

    gui = glooey.Gui(window)

    hbox = glooey.HBox()
    hbox.set_padding(5)
    widgets = {}
    scenes = callback(-1)
    for key, scene in scenes.items():
        widgets[key] = trimesh.viewer.SceneWidget(scene)
        hbox.add(widgets[key])
    gui.add(hbox)

    pyglet.clock.schedule_interval(callback, 1 / 30, widgets)
    pyglet.app.run()


if __name__ == '__main__':
    main()

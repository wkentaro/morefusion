#!/usr/bin/env python

import numpy as np
import pyglet
import trimesh
import trimesh.viewer
import trimesh.transformations as tf

import objslampp


def callback(scene):
    if not scene.play:
        return

    dataset = objslampp.datasets.YCBVideoDataset('train')
    example = dataset[scene.index]

    video_id, frame_id = dataset.ids[scene.index].split('/')

    # Reset scene for new scene.
    if video_id != scene.video_id:
        scene.geometry = {}
        scene.graph.clear()
    scene.video_id = video_id

    rgb = example['color']
    depth = example['depth']
    K = example['meta']['intrinsic_matrix']

    T_world2cam = np.r_[
        example['meta']['rotation_translation_matrix'], [[0, 0, 0, 1]]
    ]
    T_cam2world = np.linalg.inv(T_world2cam)
    pcd = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    nonnan = ~np.isnan(depth)
    geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
    scene.add_geometry(geom, transform=T_cam2world)

    # A kind of current camera view, but a bit far away to see whole scene.
    scene.camera.resolution = (rgb.shape[1], rgb.shape[0])
    scene.camera.focal = (K[0, 0], K[1, 1])
    scene.camera.transform = objslampp.extra.trimesh.to_opengl_transform(
        T_cam2world @ tf.translation_matrix([0, 0, -0.5])
    )
    # scene.set_camera()

    scene.index += 15


def main():
    scene = trimesh.Scene()
    scene.index = 0
    scene.video_id = None
    scene.play = False

    # To avoid error for empty scene.
    # We register big cube for the far parameter of rendering.
    dummy_geom = trimesh.creation.box((1, 1, 1))
    dummy_geom.visual.face_colors = (0, 0, 0, 0)
    scene.add_geometry(dummy_geom)

    scene.camera.resolution = (640, 480)

    window = trimesh.viewer.SceneViewer(
        scene, callback=callback, start_loop=False
    )

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.S:
                scene.play = not scene.play

    print('Press S key to start/pause.')

    pyglet.app.run()


if __name__ == '__main__':
    main()

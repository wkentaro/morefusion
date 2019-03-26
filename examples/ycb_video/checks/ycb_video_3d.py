#!/usr/bin/env python

import numpy as np
import pyglet
import termcolor
import trimesh
import trimesh.viewer

import objslampp


class MainApp(object):

    def __init__(self):
        self._dataset = objslampp.datasets.YCBVideoDataset(split='train')
        self._cads = {}

        scene = trimesh.Scene()
        geom = trimesh.creation.axis(origin_size=0.02)
        scene.add_geometry(geom, node_name='a', geom_name='a')

        scene.id = None
        scene.index = 0
        scene.pause = True

        window = trimesh.viewer.SceneViewer(
            scene=scene,
            callback=self._callback,
            callback_period=0.1,
            start_loop=False,
        )
        self._window = window

        @window.event
        def on_key_press(symbol, modifiers):
            key = None
            if symbol == pyglet.window.key.Q:
                key = 'q'
                window.close()
            elif symbol == pyglet.window.key.S:
                key = 's'
                scene.pause = not scene.pause
            else:
                return
            print(f'key: {key}')

        pyglet.app.run()

    def _callback(self, scene):
        if scene.pause:
            return

        image_id = self._dataset.imageset[scene.index]
        scene_id = image_id.split('/')[0]
        if scene.id and scene.id != scene_id:
            scene.graph.clear()
        scene.id = scene_id

        termcolor.cprint(f'[{scene.index:08d}] [{image_id:s}]')

        frame = self._dataset[scene.index]
        class_ids = frame['meta']['cls_indexes']
        poses = frame['meta']['poses'].transpose(2, 0, 1)

        for cls_id in frame['meta']['cls_indexes']:
            if cls_id in self._cads:
                cad = self._cads[cls_id]
            else:
                models = objslampp.datasets.YCBVideoModelsDataset()
                cad_file = models.get_model(class_id=cls_id)['textured_simple']
                cad = trimesh.load(str(cad_file), process=False)
                cad.visual = cad.visual.to_color()
                cad.visual.vertex_colors = [0.7, 0.7, 0.7]
                self._cads[cls_id] = cad

        K = frame['meta']['intrinsic_matrix']
        depth = frame['depth']
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )

        T_world2cam = frame['meta']['rotation_translation_matrix']
        T_world2cam = np.r_[T_world2cam, [[0, 0, 0, 1]]]
        T_cam2world = np.linalg.inv(T_world2cam)

        geom = trimesh.creation.axis(
            origin_size=0.02, origin_color=(1.0, 0, 0)
        )
        scene.add_geometry(
            geom, node_name='a', geom_name='a', transform=T_cam2world
        )

        isnan = np.isnan(pcd).any(axis=2)
        vertices = pcd[~isnan]
        colors = frame['color'][~isnan]
        geom = trimesh.PointCloud(vertices=vertices, color=colors)
        scene.add_geometry(
            geom, node_name='b', geom_name='b', transform=T_cam2world
        )

        for cls_id, pose in zip(class_ids, poses):
            cad = self._cads[cls_id]
            pose = np.r_[pose, [[0, 0, 0, 1]]]
            transform = T_cam2world @ pose
            scene.add_geometry(
                cad,
                node_name=f'{cls_id}',
                geom_name=f'{cls_id}',
                transform=transform,
            )
            geom = trimesh.creation.axis(origin_size=0.01)
            scene.add_geometry(
                geom,
                node_name=f'axis_{cls_id}',
                geom_name=f'axis_{cls_id}',
                transform=transform,
            )

        camera = trimesh.scene.Camera(
            resolution=(640, 480), focal=K[[0, 1], [0, 1]],
        )
        T_cam2trimesh = trimesh.transformations.rotation_matrix(
            np.deg2rad(180), [1, 0, 0]
        )
        camera.transform = T_cam2world @ T_cam2trimesh
        scene.set_camera(camera=camera)

        # scene.set_camera()  # to adjust camera location
        self._window.on_resize(None, None)  # to adjust zfar

        scene.index += 50


if __name__ == '__main__':
    import fire

    fire.Fire(MainApp)

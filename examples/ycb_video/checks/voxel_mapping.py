#!/usr/bin/env python

import imgviz
import numpy as np
import pandas
import pyglet
import termcolor
import trimesh
import trimesh.viewer

import objslampp

from tmp import ResNetFeatureExtractor


class MainApp(object):

    def __init__(self, channel='rgb', gpu=0):
        assert channel in ['rgb', 'res'], "channel must be 'rgb' or 'res'"
        assert isinstance(gpu, int), 'gpu must be integer'

        self._channel = channel

        self._dataset = objslampp.datasets.YCBVideoDataset(split='train')

        self._class_id = None
        self._mapping = objslampp.geometry.VoxelMapping(
            origin=None, pitch=None, voxel_size=32, nchannel=3
        )

        scene = trimesh.Scene()
        scene.add_geometry(
            trimesh.creation.axis(origin_size=0.01),
            node_name='a',
            geom_name='a',
        )
        window = trimesh.viewer.SceneViewer(
            scene=scene,
            callback=self._callback,
            callback_period=0.1,
            start_loop=False,
        )
        self._window = window

        scene.pause = True
        scene.index = 1000

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

        if channel == 'res':
            self._res = ResNetFeatureExtractor(gpu=gpu)

        pyglet.app.run()

    def _callback(self, scene):
        if scene.pause:
            return

        image_id = self._dataset.imageset[scene.index]

        try:
            frame = self._dataset[scene.index]
            camera_transform, pcd_roi_flat, rgb_roi_flat = \
                self._process_frame(frame)
        except Exception as e:
            print('An error occurs. Stopping..')
            print(e)
            scene.pause = True
            return

        termcolor.cprint(f'[{scene.index}] {image_id}', attrs={'bold': True})

        if 0:
            # point cloud
            geom = trimesh.PointCloud(
                vertices=pcd_roi_flat, color=rgb_roi_flat
            )
            scene.add_geometry(geom)

        # voxel origin
        geom = trimesh.creation.axis(origin_size=0.01)
        geom.apply_translation(self._mapping.origin)
        if 'a' in scene.geometry:
            scene.geometry['a'] = geom
        else:
            scene.add_geometry(geom, node_name='a', geom_name='a')

        geom = trimesh.creation.axis(origin_size=0.01)
        geom.apply_transform(camera_transform)
        scene.add_geometry(geom)

        # voxel
        geom = self._mapping.as_boxes()
        if 'b' in scene.geometry:
            scene.geometry['b'] = geom
        else:
            scene.add_geometry(geom, node_name='b', geom_name='b')

        # voxel bbox
        if 'c' not in scene.geometry:
            geom = self._mapping.as_bbox()
            scene.add_geometry(geom, node_name='c', geom_name='c')

        scene.set_camera()  # to adjust camera location
        self._window.on_resize(None, None)  # to adjust zfar

        scene.index += 50

    def _process_frame(self, frame):
        meta = frame['meta']
        K = meta['intrinsic_matrix']
        rgb = frame['color']
        depth = frame['depth']
        label = frame['label']

        if self._class_id is None:
            self._class_id = meta['cls_indexes'][2]
            print(f'Initialized class_id: {self._class_id}')

            assert self._mapping.pitch is None
            df = pandas.read_csv('data/voxel_size.csv')
            pitch = float(df['voxel_size'][df['class_id'] == self._class_id])
            self._mapping.pitch = pitch
            print(f'Initialized pitch: {self._mapping.pitch}')

        mask = label == self._class_id
        bbox = imgviz.instances.mask_to_bbox([mask])[0]
        y1, x1, y2, x2 = bbox.round().astype(int)
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2]
        )

        if self._channel == 'rgb':
            rgb_roi = rgb[mask]
        else:
            assert self._channel == 'res'
            feat = self._res.extract_feature(rgb)
            feat_viz = self._res.feature2rgb(feat, label != 0)
            rgb_roi = feat_viz[mask]
        rgb_roi_flat = rgb_roi.reshape(-1, 3)

        pcd_roi = pcd[mask]
        pcd_roi_flat = pcd_roi.reshape(-1, 3)

        # remove nan
        keep = ~np.isnan(pcd_roi_flat).any(axis=1)
        rgb_roi_flat = rgb_roi_flat[keep]
        pcd_roi_flat = pcd_roi_flat[keep]

        T = meta['rotation_translation_matrix']
        T = np.r_[T, [[0, 0, 0, 1]]]
        camera_transform = np.linalg.inv(T)
        # camera frame -> world frame
        pcd_roi_flat = trimesh.transform_points(
            pcd_roi_flat, camera_transform
        )

        if self._mapping.origin is None:
            aabb_min, aabb_max = objslampp.geometry.get_aabb_from_points(
                pcd_roi_flat
            )
            aabb_extents = aabb_max - aabb_min
            aabb_center = aabb_extents / 2 + aabb_min
            origin = aabb_center - self._mapping.voxel_bbox_extents / 2
            self._mapping.origin = origin
        self._mapping.add(pcd_roi_flat, rgb_roi_flat.astype(float) / 255)

        return camera_transform, pcd_roi_flat, rgb_roi_flat


if __name__ == '__main__':
    import fire

    fire.Fire(MainApp)

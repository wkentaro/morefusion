#!/usr/bin/env python

import imgviz
import numpy as np
import open3d
import pyglet
import scipy.cluster.hierarchy
import termcolor
import trimesh
import trimesh.viewer

import objslampp


def get_aabb_points(points):
    # getting aabb
    with objslampp.utils.timer('voxel_down_sample'):
        pcd_roi_flat_down = voxel_down_sample(
            points=points, voxel_size=0.01
        )
    with objslampp.utils.timer('euclidean_clustering'):
        fclusterdata = scipy.cluster.hierarchy.fclusterdata(
            pcd_roi_flat_down, criterion='distance', t=0.02
        )
    cluster_ids, cluster_counts = np.unique(
        fclusterdata, return_counts=True
    )
    cluster_id = cluster_ids[np.argmax(cluster_counts)]
    keep = fclusterdata == cluster_id
    pcd_roi_flat_down = pcd_roi_flat_down[keep]
    aabb_min = pcd_roi_flat_down.min(axis=0)
    aabb_max = pcd_roi_flat_down.max(axis=0)
    return aabb_min, aabb_max


def voxel_down_sample(points, voxel_size):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    pcd = open3d.voxel_down_sample(pcd, voxel_size=voxel_size)
    dst_points = np.asarray(pcd.points)
    return dst_points


class MainApp(object):

    pitch = 4. / 512  # 0.0078125
    voxel_size = 32
    D = 3  # RGB

    def __init__(self):
        self.dataset = objslampp.datasets.YCBVideoDataset()

        self.class_id = None
        self.origin = None
        self.matrix = None
        self.matrix_values = None
        self.camera_transform = None

        scene = trimesh.Scene()
        scene.add_geometry(
            trimesh.creation.axis(origin_size=0.01),
            node_name='a',
            geom_name='a',
        )
        window = trimesh.viewer.SceneViewer(
            scene=scene,
            callback=self.callback,
            callback_period=0.1,
            start_loop=False,
        )
        self.window = window

        scene.pause = True
        scene.index = 1000

        @window.event
        def on_key_press(symbol, modifiers):
            key = symbol
            if symbol == pyglet.window.key.Q:
                key = 'q'
                window.close()
            elif symbol == pyglet.window.key.S:
                key = 's'
                scene.pause = not scene.pause
            print(f'key: {key}')

    def run(self):
        pyglet.app.run()

    @property
    def voxel_bbox_extents(self):
        return np.array((self.voxel_size * self.pitch,) * 3, dtype=float)

    def callback(self, scene):
        if scene.pause:
            return

        image_id = self.dataset.imageset('train')[scene.index]

        try:
            frame = self.dataset.get_frame(image_id)
            pcd_roi_flat, rgb_roi_flat = self.process_frame(frame)
        except Exception:
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
        geom.apply_translation(self.origin)
        if 'a' in scene.geometry:
            scene.geometry['a'] = geom
        else:
            scene.add_geometry(geom, node_name='a', geom_name='a')
        # scene.add_geometry(geom, node_name='a', geom_name='a')

        geom = trimesh.creation.axis(origin_size=0.01)
        geom.apply_transform(self.camera_transform)
        scene.add_geometry(geom)

        # voxel
        geom = trimesh.voxel.Voxel(self.matrix, self.pitch, self.origin)
        geom = geom.as_boxes()
        I, J, K = zip(*np.argwhere(self.matrix))
        geom.visual.face_colors = \
            self.matrix_values[I, J, K].repeat(12, axis=0)
        if 'b' in scene.geometry:
            scene.geometry['b'] = geom
        else:
            scene.add_geometry(geom, node_name='b', geom_name='b')

        # voxel bbox
        if 'c' not in scene.geometry:
            geom = objslampp.vis.trimesh.wired_box(
                self.voxel_bbox_extents,
                translation=self.origin + self.voxel_bbox_extents / 2,
            )
            scene.add_geometry(geom, node_name='c', geom_name='c')

        scene.set_camera()  # to adjust camera location
        self.window.on_resize(None, None)  # to adjust zfar

        scene.index += 15

    def process_frame(self, frame):
        meta = frame['meta']
        K = meta['intrinsic_matrix']
        rgb = frame['color']
        depth = frame['depth']
        label = frame['label']

        if self.class_id is None:
            self.class_id = meta['cls_indexes'][2]

        mask = frame['label'] == self.class_id
        bbox = imgviz.instances.mask_to_bbox([mask])[0]
        y1, x1, y2, x2 = bbox.round().astype(int)
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2]
        )

        rgb_roi = rgb[mask]
        rgb_roi_flat = rgb_roi.reshape(-1, 3)
        pcd_roi = pcd[mask]
        pcd_roi_flat = pcd_roi.reshape(-1, 3)

        # remove nan
        keep = ~np.isnan(pcd_roi_flat).any(axis=1)
        rgb_roi_flat = rgb_roi_flat[keep]
        pcd_roi_flat = pcd_roi_flat[keep]

        T = meta['rotation_translation_matrix']
        T = np.r_[T, [[0, 0, 0, 1]]]
        self.camera_transform = np.linalg.inv(T)
        # camera frame -> world frame
        pcd_roi_flat = trimesh.transform_points(
            pcd_roi_flat, self.camera_transform
        )

        aabb_min, aabb_max = get_aabb_points(pcd_roi_flat)

        # TODO(wkentaro): origin should be adjusted while scan
        if self.origin is None:
            aabb_extents = aabb_max - aabb_min
            aabb_center = aabb_extents / 2 + aabb_min
            self.origin = aabb_center - self.voxel_bbox_extents / 2
            assert self.matrix is None
            assert self.matrix_values is None
            self.matrix = np.zeros((self.voxel_size,) * 3, dtype=bool)
            self.matrix_values = np.zeros(
                (self.voxel_size,) * 3 + (self.D,), dtype=float
            )

        indices = trimesh.voxel.points_to_indices(
            pcd_roi_flat, self.pitch, self.origin
        )
        keep = ((indices >= 0) & (indices < self.voxel_size)).all(axis=1)
        indices = indices[keep]
        I, J, K = zip(*indices)
        self.matrix[I, J, K] = True
        # TODO(wkentaro): replace with average or max of feature
        self.matrix_values[I, J, K] = rgb_roi_flat[keep].astype(float) / 255

        return pcd_roi_flat, rgb_roi_flat


if __name__ == '__main__':
    MainApp().run()

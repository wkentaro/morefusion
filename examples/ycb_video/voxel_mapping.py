#!/usr/bin/env python

from chainer.backends import cuda
from chainercv.links.model.resnet import ResNet50
import imgviz
import numpy as np
import open3d
import scipy.cluster.hierarchy
import trimesh

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
    # print('cluster_ids:', cluster_ids, 'cluster_counts:', cluster_counts)
    cluster_id = cluster_ids[np.argmax(cluster_counts)]
    keep = fclusterdata == cluster_id
    # print('pcd_roi_flat:', pcd_roi_flat.shape)
    pcd_roi_flat_down = pcd_roi_flat_down[keep]
    # print('pcd_roi_flat_down:', pcd_roi_flat_down.shape)
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

        self.scene = None

        self.class_id = None
        self.origin = None
        self.matrix = None
        self.matrix_values = None

        cuda.get_device_from_id(0).use()
        self.resnet = ResNet50(pretrained_model='imagenet', arch='he')
        self.resnet.pick = ['res4']
        self.resnet.remove_unused()
        self.resnet.to_gpu()

        self.nchannel2rgb = imgviz.Nchannel2RGB()

    @property
    def voxel_bbox_extents(self):
        return np.array((self.voxel_size * self.pitch,) * 3, dtype=float)

    @property
    def image_ids(self):
        image_id1 = self.dataset.imageset('train')[1000]
        image_id2 = self.dataset.imageset('train')[1500]
        return [image_id1, image_id2]

    def extract_feature(self, rgb):
        x = rgb.transpose(2, 0, 1)
        x = x - self.resnet.mean
        x = x[None]
        x = cuda.to_gpu(x)
        feat, = self.resnet(x)
        feat = cuda.to_cpu(feat[0].array)
        return feat.transpose(1, 2, 0)

    def feature2rgb(self, feat, mask_fg):
        dst = self.nchannel2rgb(feat, dtype=float)
        H, W = mask_fg.shape[:2]
        dst = imgviz.resize(dst, height=H, width=W)
        dst = (dst * 255).astype(np.uint8)
        dst[~mask_fg] = 0
        return dst

    def run(self):
        # world coordinates
        self.scene = trimesh.Scene()

        for image_id in self.image_ids:
            frame = self.dataset.get_frame(image_id)
            self.process_frame(frame)

            feat = self.extract_feature(frame['color'])
            feat_viz = self.feature2rgb(feat, frame['label'] != 0)
            rgb_roi_viz = frame['color'].copy()
            rgb_roi_viz[frame['label'] != self.class_id] = 0
            feat_roi_viz = feat_viz.copy()
            feat_roi_viz[frame['label'] != self.class_id] = 0
            viz = imgviz.tile(
                [frame['color'], feat_viz, rgb_roi_viz, feat_roi_viz],
                border=(255, 255, 255),
            )
            imgviz.io.pyglet_imshow(viz)

            self.show(show_voxel=1)

    def show(self, show_voxel=True, **kwargs):
        scene = self.scene.copy()

        if show_voxel:
            # voxel origin
            geom = trimesh.creation.axis()
            geom.apply_translation(self.origin)
            scene.add_geometry(geom)

            # voxel
            geom = trimesh.voxel.Voxel(self.matrix, self.pitch, self.origin)
            geom = geom.as_boxes()
            I, J, K = zip(*np.argwhere(self.matrix))
            geom.visual.face_colors = \
                self.matrix_values[I, J, K].repeat(12, axis=0)
            scene.add_geometry(geom)

            # voxel bbox
            geom = objslampp.vis.trimesh.wired_box(
                self.voxel_bbox_extents,
                translation=self.origin + self.voxel_bbox_extents / 2,
            )
            scene.add_geometry(geom)

        scene.show(**kwargs)

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

        if 0:
            rgb_roi = rgb[mask]
        else:
            feat = self.extract_feature(rgb)
            feat_viz = self.feature2rgb(feat, label != 0)
            rgb_roi = feat_viz[mask]

        rgb_roi_flat = rgb_roi.reshape(-1, 3)
        pcd_roi = pcd[mask]
        pcd_roi_flat = pcd_roi.reshape(-1, 3)

        # remove nan
        keep = ~np.isnan(pcd_roi_flat).any(axis=1)
        rgb_roi_flat = rgb_roi_flat[keep]
        pcd_roi_flat = pcd_roi_flat[keep]

        # world -> camera
        T = meta['rotation_translation_matrix']
        T = np.r_[T, [[0, 0, 0, 1]]]
        pcd_roi_flat = trimesh.transform_points(pcd_roi_flat, np.linalg.inv(T))

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

        # ---------------------------------------------------------------------

        geom = trimesh.PointCloud(vertices=pcd_roi_flat, color=rgb_roi_flat)
        self.scene.add_geometry(geom)

        geom = trimesh.creation.axis(origin_size=0.02)
        geom.apply_transform(np.linalg.inv(T))
        self.scene.add_geometry(geom)

        extents = aabb_max - aabb_min
        geom = objslampp.vis.trimesh.wired_box(
            extents=extents,
            translation=extents / 2 + aabb_min,
        )
        self.scene.add_geometry(geom)


if __name__ == '__main__':
    MainApp().run()

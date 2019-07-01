#!/usr/bin/env python

import chainer
import imgviz
import numpy as np
import trimesh
import trimesh.transformations as tf

import objslampp

import preliminary


class MultiInstanceICPRegistration:

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
        self._rgb = rgb
        self._pcd = pcd
        self._instance_label = instance_label
        self._instance_label_viz = imgviz.label2rgb(instance_label)
        self._instance_ids = instance_ids
        self._class_ids = dict(zip(instance_ids, class_ids))
        self._Ts_cad2cam_true = dict(zip(instance_ids, Ts_cad2cam_true))

        self._cads = {}
        for instance_id in self._instance_ids:
            class_id = self._class_ids[instance_id]
            cad_file = self._models.get_cad_file(class_id=class_id)
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

    def register_instance(self, instance_id):
        pcd = self._pcd
        nonnan = ~np.isnan(pcd).any(axis=2)
        mask = self._instance_label == instance_id
        pcd_depth = pcd[nonnan & mask]

        class_id = self._class_ids[instance_id]
        pcd_file = self._models.get_pcd_file(class_id=class_id)
        pcd_cad = np.loadtxt(pcd_file, dtype=np.float32)

        self._instance_id = instance_id
        self._pcd_cad = pcd_cad
        self._pcd_depth = pcd_depth

        registration = preliminary.ICPRegistration(
            pcd_depth, pcd_cad, self._Ts_cad2cam_pred[instance_id]
        )
        with chainer.using_config('debug', True):
            for transform in registration.register_iterative(
                iteration=100, voxel_size=0.01
            ):
                self._Ts_cad2cam_pred[instance_id] = transform
                yield

    def visualize(self):
        rgb = self._rgb
        pcd = self._pcd

        instance_id = self._instance_id
        T_cad2cam_true = self._Ts_cad2cam_true[instance_id]

        cad = self._cads[instance_id]
        T_cad2cam_pred = self._Ts_cad2cam_pred[instance_id]

        camera = trimesh.scene.Camera(
            resolution=(640, 480),
            fov=(60 * 0.7, 45 * 0.7),
            transform=objslampp.extra.trimesh.camera_transform(),
        )

        scenes = {}

        scenes['pcd'] = trimesh.Scene(camera=camera)
        nonnan = ~np.isnan(pcd).any(axis=2)
        geom = trimesh.PointCloud(pcd[nonnan], colors=rgb[nonnan])
        scenes['pcd'].add_geometry(geom)

        scenes['instance_points'] = trimesh.Scene(camera=camera)
        # points_source: points_from_depth
        geom = trimesh.PointCloud(
            vertices=self._pcd_depth, colors=(0, 1., 0)
        )
        scenes['instance_points'].add_geometry(geom, geom_name='points_source')
        # points_target: points_from_cad
        geom = trimesh.PointCloud(
            vertices=self._pcd_cad, colors=(1., 0, 0)
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
        scenes['scene_cad'] = trimesh.Scene(camera=camera)
        scenes['scene_pcd'] = trimesh.Scene(camera=camera)
        nonnan = ~np.isnan(pcd).any(axis=2)
        geom = trimesh.PointCloud(pcd[nonnan], colors=rgb[nonnan])
        scenes['scene_pcd'].add_geometry(geom, geom_name='pcd')
        for instance_id in self._instance_ids:
            if instance_id not in self._Ts_cad2cam_pred:
                continue
            assert instance_id in self._cads
            T_cad2cam_pred = self._Ts_cad2cam_pred[instance_id]
            for key in ['scene_cad', 'scene_pcd']:
                scenes[key].add_geometry(
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


def refinement(
    instance_ids,
    class_ids,
    rgb,
    pcd,
    instance_label,
    Ts_cad2cam_true,
    Ts_cad2cam_pred=None,
):
    registration = MultiInstanceICPRegistration(
        rgb=rgb,
        pcd=pcd,
        instance_label=instance_label,
        instance_ids=instance_ids,
        class_ids=class_ids,
        Ts_cad2cam_true=Ts_cad2cam_true,
        Ts_cad2cam_pred=Ts_cad2cam_pred,
    )

    coms = np.array([
        np.nanmedian(pcd[instance_label == i], axis=0) for i in instance_ids
    ])
    instance_ids = np.array(instance_ids)[np.argsort(coms[:, 2])]
    instance_ids = iter(instance_ids)

    # -------------------------------------------------------------------------

    def scenes_ggroup():
        for instance_id in instance_ids:
            yield (
                registration.visualize()
                for _ in registration.register_instance(instance_id)
            )

    objslampp.extra.trimesh.display_scenes(
        scenes_ggroup(), height=360, width=480, tile=(2, 3))

    return registration


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

    Ts_cad2cam_pred = np.zeros_like(Ts_cad2cam_true)
    for i, ins_id in enumerate(instance_ids):
        mask = instance_label == ins_id
        centroid = np.nanmean(pcd[mask], axis=0)
        Ts_cad2cam_pred[i] = tf.translation_matrix(centroid)

    return refinement(
        instance_ids,
        class_ids,
        rgb,
        pcd,
        instance_label,
        Ts_cad2cam_true,
        Ts_cad2cam_pred,
    )


if __name__ == '__main__':
    main()

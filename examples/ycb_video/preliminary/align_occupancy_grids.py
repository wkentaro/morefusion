#!/usr/bin/env python

import imgviz
import numpy as np
import trimesh
import trimesh.transformations as tf

import objslampp

import preliminary


class MultiInstanceOccupancyRegistration:

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
        self._mapping = preliminary.MultiInstanceOctreeMapping()

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

    def update_octree(self, instance_id):
        T_cad2cam_pred = self._Ts_cad2cam_pred[instance_id]

        class_id = self._class_ids[instance_id]
        pcd_file = self._models.get_pcd_file(class_id=class_id)
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
        cad_file = models.get_cad_file(class_id=class_id)
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
        pcd_file = models.get_pcd_file(class_id=class_id)
        points_source = np.loadtxt(pcd_file, dtype=np.float32)
        points_source = objslampp.extra.open3d.voxel_down_sample(
            points_source, voxel_size=pitch
        )
        points_source = points_source.astype(np.float32)

        self._instance_id = instance_id

        grids = np.stack((grid_target, grid_nontarget, grid_empty))
        grids = grids.astype(np.float32)
        registration = preliminary.OccupancyRegistration(
            points_source,
            grids,
            pitch=pitch,
            origin=origin,
            threshold=threshold,
            transform_init=self._Ts_cad2cam_pred[instance_id],
        )
        self._grids = grids
        self._pitch = pitch
        self._origin = origin

        for T_cad2cam in registration.register_iterative(100):
            self._Ts_cad2cam_pred[instance_id] = T_cad2cam
            yield

    def visualize(self):  # NOQA
        # scene-level
        rgb = self._rgb
        pcd = self._pcd

        scenes = {}

        # ---------------------------------------------------------------------

        # instance-level
        instance_id = self._instance_id
        cad = self._cads[instance_id]
        T_cad2cam_true = self._Ts_cad2cam_true[instance_id]
        T_cad2cam_pred = self._Ts_cad2cam_pred[instance_id]
        grid_target = self._grids
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

        # ---------------------------------------------------------------------

        # scene_pcd
        scenes['scene_pcd_only'] = trimesh.Scene()
        scenes['scene_cad'] = trimesh.Scene()
        scenes['scene_pcd'] = trimesh.Scene()
        nonnan = ~np.isnan(pcd).any(axis=2)
        geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
        scenes['scene_pcd_only'].add_geometry(geom, geom_name='pcd')
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

        # set camera
        camera = trimesh.scene.Camera(
            resolution=(640, 480),
            fov=(60 * 0.7, 45 * 0.7),
            transform=objslampp.extra.trimesh.camera_transform(),
        )
        for scene in scenes.values():
            scene.camera = camera
        return scenes


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
    registration = MultiInstanceOccupancyRegistration(
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

    def scenes_ggroup():
        for ins_id in instance_ids:
            yield (
                registration.visualize()
                for _ in registration.register_instance(ins_id)
            )

    preliminary.display_scenes(
        scenes_ggroup(),
        height=int(480 * 0.6),
        width=int(640 * 0.6),
        tile=(2, 4),
    )

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

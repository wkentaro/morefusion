#!/usr/bin/env python

import numpy as np
import trimesh

import objslampp

import preliminary


def main():
    dataset = objslampp.datasets.YCBVideoDataset('train')
    frame = dataset[0]

    class_ids = frame['meta']['cls_indexes']
    instance_ids = class_ids
    K = frame['meta']['intrinsic_matrix']
    rgb = frame['color']
    depth = frame['depth']
    height, width = rgb.shape[:2]
    pcd = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    instance_label = frame['label']

    instance_ids_all = np.r_[0, instance_ids]

    # build octrees
    pitch = 0.005
    mapping = preliminary.MultiInstanceOctreeMapping()
    for ins_id in instance_ids_all:
        mask = instance_label == ins_id
        mapping.initialize(ins_id, pitch=pitch)
        mapping.integrate(ins_id, mask, pcd)

    target_id = 8
    mask = instance_label == target_id

    centroid = np.nanmean(pcd[mask], axis=0)
    models = objslampp.datasets.YCBVideoModels()
    diagonal = models.get_bbox_diagonal(models.get_cad_model(target_id))
    aabb_min = centroid - diagonal / 2
    aabb_max = aabb_min + diagonal
    print(aabb_min, aabb_max)

    occupied_t, empty_i = mapping.get_target_pcds(
        target_id, aabb_min, aabb_max
    )
    occupied_u = []
    empty = [empty_i]
    for ins_id in instance_ids_all:
        if ins_id == target_id:
            continue
        occupied_u_i, empty_i = mapping.get_target_pcds(
            ins_id, aabb_min, aabb_max
        )
        occupied_u.append(occupied_u_i)
        empty.append(empty_i)
    occupied_u = np.concatenate(occupied_u, axis=0)
    empty = np.concatenate(empty, axis=0)

    print(occupied_t.shape)
    print(occupied_u.shape)
    print(empty.shape)

    camera = trimesh.scene.Camera(
        resolution=(640, 480),
        focal=(K[0, 0], K[1, 1]),
        transform=objslampp.extra.trimesh.to_opengl_transform(),
    )
    scene = trimesh.Scene(camera=camera)
    geom = trimesh.PointCloud(vertices=occupied_t, colors=(1., 0, 0))
    scene.add_geometry(geom)
    geom = trimesh.PointCloud(vertices=occupied_u, colors=(0, 1., 0))
    scene.add_geometry(geom)
    geom = trimesh.PointCloud(vertices=empty, colors=(0.5, 0.5, 0.5, 0.5))
    scene.add_geometry(geom)
    geom = trimesh.path.creation.box_outline((aabb_max - aabb_min))
    geom.apply_translation(centroid)
    scene.add_geometry(geom)
    preliminary.display_scenes({__file__: scene})


if __name__ == '__main__':
    main()

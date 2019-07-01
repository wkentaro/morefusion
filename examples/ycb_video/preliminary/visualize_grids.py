#!/usr/bin/env python

import numpy as np
import trimesh
import trimesh.transformations as tf

import objslampp

import preliminary


def visualize_grids(
    camera,
    rgb,
    pcd,
    pitch,
    origin,
    grid_target,
    grid_nontarget,
    grid_empty,
):
    scenes = {}

    scene_common = trimesh.Scene(camera=camera)
    # point_cloud
    nonnan = ~np.isnan(pcd).any(axis=2)
    geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
    scene_common.add_geometry(geom, geom_name='point_cloud')
    # grid_aabb
    dimensions = np.array(grid_target.shape)
    center = origin + dimensions / 2 * pitch
    geom = trimesh.path.creation.box_outline(
        extents=dimensions * pitch,
        transform=tf.translation_matrix(center),
    )
    scene_common.add_geometry(geom, geom_name='grid_aabb')
    # camera
    camera_marker = camera.copy()
    camera_marker.transform = \
        objslampp.extra.trimesh.from_opengl_transform(camera.transform)
    geom = trimesh.creation.camera_marker(camera_marker, marker_height=0.1)
    scene_common.add_geometry(geom, geom_name='camera_marker')

    scenes['occupied'] = trimesh.Scene(
        camera=scene_common.camera,
        geometry=scene_common.geometry,
    )
    # grid_target
    voxel = trimesh.voxel.Voxel(
        matrix=grid_target,
        pitch=pitch,
        origin=origin,
    )
    geom = voxel.as_boxes(colors=(1., 0, 0, 0.5))
    scenes['occupied'].add_geometry(geom, geom_name='grid_target')
    # grid_nontarget
    voxel = trimesh.voxel.Voxel(
        matrix=grid_nontarget,
        pitch=pitch,
        origin=origin,
    )
    geom = voxel.as_boxes(colors=(0, 1., 0, 0.5))
    scenes['occupied'].add_geometry(geom, geom_name='grid_nontarget')

    scenes['empty'] = trimesh.Scene(
        camera=scene_common.camera,
        geometry=scene_common.geometry,
    )
    # grid_empty
    voxel = trimesh.voxel.Voxel(
        matrix=grid_empty,
        pitch=pitch,
        origin=origin,
    )
    geom = voxel.as_boxes(colors=(0.5, 0.5, 0.5, 0.5))
    scenes['empty'].add_geometry(geom, geom_name='grid_empty')

    return scenes


def main():
    dataset = objslampp.datasets.YCBVideoDataset('train')
    frame = dataset.get_example(1000)

    # scene-level data
    class_ids = frame['meta']['cls_indexes']
    instance_ids = class_ids
    Ts_cad2cam_true = np.tile(np.eye(4), (len(instance_ids), 1, 1))
    Ts_cad2cam_true[:, :3, :4] = frame['meta']['poses'].transpose(2, 0, 1)
    K = frame['meta']['intrinsic_matrix']
    rgb = frame['color']
    nonnan = ~np.isnan(frame['depth'])
    pcd = objslampp.geometry.pointcloud_from_depth(
        frame['depth'], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    instance_label = frame['label']

    mapping = preliminary.MultiInstanceOctreeMapping()

    # background
    ins_id = 0
    assert (ins_id == instance_ids).sum() == 0
    mapping.initialize(ins_id, pitch=0.01)
    mask = instance_label == 0
    mapping.integrate(ins_id, mask, pcd)

    for ins_id in instance_ids:
        mapping.initialize(ins_id, pitch=0.01)

        mask = instance_label == ins_id
        mapping.integrate(ins_id, mask, pcd)

    # import imgviz
    # imgviz.io.pyglet_imshow(rgb)
    # imgviz.io.pyglet_run()

    camera = trimesh.scene.Camera(
        resolution=(640, 480),
        focal=(K[0, 0] * 0.6, K[1, 1] * 0.6),
        transform=objslampp.extra.trimesh.to_opengl_transform(),
    )
    all_scenes = {}
    for ins_id, cls_id in zip(instance_ids, class_ids):
        class_name = objslampp.datasets.ycb_video.class_names[cls_id]

        pitch = 0.01
        dimensions = np.array([20, 20, 20], dtype=int)
        mask = instance_label == ins_id
        centroid = np.nanmean(pcd[mask & nonnan], axis=0)
        origin = centroid - ((dimensions / 2 - 0.5) * pitch)

        grid_target, grid_nontarget, grid_empty = mapping.get_target_grids(
            ins_id, origin=origin, pitch=pitch, dimensions=dimensions
        )

        # ---------------------------------------------------------------------

        scenes = visualize_grids(
            camera,
            rgb,
            pcd,
            pitch,
            origin,
            grid_target,
            grid_nontarget,
            grid_empty,
        )

        for name in list(scenes.keys()):
            new_name = f'{ins_id} : {class_name} : {name}'
            scenes[new_name] = scenes.pop(name)
        all_scenes.update(scenes)

    objslampp.extra.trimesh.display_scenes(
        all_scenes,
        height=int(480 * 0.5),
        width=int(640 * 0.5),
        tile=(len(instance_ids), 2),
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import imgviz
import numpy as np
import trimesh

import morefusion


def visualize_pcds(
    camera, camera_transform, instance_ids, pcds_occupied, pcds_empty,
):
    scenes = {}

    scene_common = trimesh.Scene(camera=camera)
    # camera
    camera_marker = camera.copy()
    camera_marker.transform = morefusion.extra.trimesh.from_opengl_transform(
        camera_transform
    )
    geom = trimesh.creation.camera_marker(camera_marker, marker_height=0.1)
    scene_common.add_geometry(geom, geom_name="camera_marker")

    scenes["occupied"] = trimesh.Scene(
        camera=camera,
        geometry=scene_common.geometry,
        camera_transform=camera_transform,
    )
    scenes["empty"] = trimesh.Scene(
        camera=camera,
        geometry=scene_common.geometry,
        camera_transform=camera_transform,
    )
    colormap = imgviz.label_colormap()
    for ins_id, occupied, empty in zip(
        instance_ids, pcds_occupied, pcds_empty
    ):
        geom = trimesh.PointCloud(vertices=occupied, colors=colormap[ins_id])
        scenes["occupied"].add_geometry(geom)
        geom = trimesh.PointCloud(vertices=empty, colors=[0.5, 0.5, 0.5])
        scenes["empty"].add_geometry(geom)

    return scenes


def main():
    dataset = morefusion.datasets.YCBVideoDataset("train")
    frame = dataset[0]

    class_ids = frame["meta"]["cls_indexes"]
    instance_ids = class_ids
    K = frame["meta"]["intrinsic_matrix"]
    rgb = frame["color"]
    depth = frame["depth"]
    height, width = rgb.shape[:2]
    pcd = morefusion.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    instance_label = frame["label"]

    instance_ids_all = np.r_[0, instance_ids]

    # build octrees
    pitch = 0.005
    mapping = morefusion.contrib.MultiInstanceOctreeMapping()
    for ins_id in instance_ids_all:
        mask = instance_label == ins_id
        mapping.initialize(ins_id, pitch=pitch)
        mapping.integrate(ins_id, mask, pcd)

    camera = trimesh.scene.Camera(
        resolution=(640, 480), focal=(K[0, 0], K[1, 1]),
    )
    camera_transform = morefusion.extra.trimesh.to_opengl_transform()

    pcds_occupied = []
    pcds_empty = []
    for ins_id in instance_ids_all:
        occupied, empty = mapping.get_target_pcds(ins_id)
        pcds_occupied.append(occupied)
        pcds_empty.append(empty)

    scenes = visualize_pcds(
        camera, camera_transform, instance_ids_all, pcds_occupied, pcds_empty,
    )
    morefusion.extra.trimesh.display_scenes(scenes, tile=(1, 2))


if __name__ == "__main__":
    main()

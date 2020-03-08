#!/usr/bin/env python

import numpy as np
import trimesh
import trimesh.transformations as tf

import morefusion


dataset = morefusion.datasets.YCBVideoDataset(split="train")

pitch = 0.001
voxel_dim = 128

mapping = None
video_id = None
class_id = None
for index in range(0, len(dataset), 100):
    image_id = dataset.ids[index]

    print(f"[{index:08d}] [{image_id}]")

    video_id_current = image_id.split("/")[0]
    if video_id is None:
        video_id = video_id_current
    elif video_id != video_id_current:
        break

    example = dataset[index]

    class_ids = example["meta"]["cls_indexes"]
    if class_id is None:
        class_id = class_ids[0]

    rgb = example["color"]
    depth = example["depth"]
    K = example["meta"]["intrinsic_matrix"]
    mask = example["label"] == class_id

    pcd = morefusion.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
    )
    mask = ~np.isnan(pcd).any(axis=2) & mask

    points = pcd[mask]
    values = rgb[mask]

    T_world2camera = np.r_[
        example["meta"]["rotation_translation_matrix"], [[0, 0, 0, 1]],
    ]
    T_camera2world = tf.inverse_matrix(T_world2camera)

    # camera frame -> world frame
    points = trimesh.transform_points(points, T_camera2world)

    if mapping is None:
        centroid = points.mean(axis=0)
        origin = centroid - pitch * voxel_dim / 2
        mapping = morefusion.geometry.VoxelMapping(
            origin=origin, pitch=pitch, voxel_dim=voxel_dim, nchannel=3
        )

    mapping.add(points, values)

print("pitch:", pitch)
print("voxel_dim:", voxel_dim)
print("class_id:", class_id)

scene = trimesh.Scene()
boxes = mapping.as_boxes()
scene.add_geometry(boxes)
box = mapping.as_bbox()
scene.add_geometry(box)
morefusion.extra.trimesh.display_scenes({"scene": scene}, rotate=True)

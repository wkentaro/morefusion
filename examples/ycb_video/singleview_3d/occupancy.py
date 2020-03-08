#!/usr/bin/env python

import argparse
import sys

import numpy as np
import trimesh
import trimesh.transformations as tf

import morefusion

from common import Inference

sys.path.insert(0, "../preliminary")  # NOQA
from align_occupancy_grids import refinement


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--prior", action="store_true", help="prior of ground and bin"
)
args = parser.parse_args()

models = morefusion.datasets.YCBVideoModels()
dataset = "my_real"
inference = Inference(dataset=dataset, gpu=0)
frame, Ts_cad2cam_true, Ts_cad2cam_pred = inference(index=0, bg_class=True)

points_occupied = {}
if args.prior:
    # ground
    dim = (64, 64, 16)
    pitch = 0.01
    matrix = np.ones(dim, dtype=bool)
    origin = -dim[0] * pitch / 2, -dim[1] * pitch / 2, -dim[2] * pitch
    points = trimesh.voxel.matrix_to_points(matrix, pitch, origin)
    if dataset == "my_synthetic":
        points = trimesh.transform_points(
            points, frame["Ts_cad2cam"][frame["instance_ids"] == 0][0]
        )
        points_occupied[0] = points
        # bin
        mesh = trimesh.load(str(frame["cad_files"][1]))
        mesh.apply_transform(
            frame["Ts_cad2cam"][frame["instance_ids"] == 1][0]
        )
        points_occupied[1] = mesh.voxelized(0.01).points
    else:
        T = tf.translation_matrix([0, 0, 0.70])
        R = tf.rotation_matrix(np.deg2rad(-15), [1, 0, 0], [0, 0, 0])
        R = tf.rotation_matrix(np.deg2rad(2), [0, 0, 1], [0, 0, 0]) @ R
        points = trimesh.transform_points(points, T @ R)
        points_occupied[0] = points
        # mesh = morefusion.extra.trimesh.bin_model(
        #     (0.38, 0.25, 0.15), thickness=0.03)
        # T2 = tf.translation_matrix([0.0, 0, 0.45])
        # R_flip = tf.rotation_matrix(np.deg2rad(180), [0, 1, 0], [0, 0, 0])
        # mesh.apply_transform(T2 @ R @ R_flip)
        # points_occupied[0] = np.vstack((points, mesh.voxelized(0.01).points))
        # isnan = np.isnan(frame['pcd']).any(axis=2)
        # scene = trimesh.Scene()
        # scene.add_geometry(mesh)
        # scene.add_geometry(
        #     trimesh.PointCloud(
        #         vertices=frame['pcd'][~isnan], colors=frame['rgb'][~isnan]))
        # scene.show()

keep = np.isin(frame["class_ids"], inference.dataset._class_ids)
frame["class_ids"] = frame["class_ids"][keep]
frame["instance_ids"] = frame["instance_ids"][keep]
frame["Ts_cad2cam"] = frame["Ts_cad2cam"][keep]

refinement(
    instance_ids=frame["instance_ids"],
    class_ids=frame["class_ids"],
    rgb=frame["rgb"],
    pcd=frame["pcd"],
    instance_label=frame["instance_label"],
    Ts_cad2cam_true=frame["Ts_cad2cam"],
    Ts_cad2cam_pred=Ts_cad2cam_pred,
    points_occupied=points_occupied,
)

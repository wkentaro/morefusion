#!/usr/bin/env python

import imgviz
import numpy as np
import trimesh
import trimesh.transformations as ttf

import morefusion

import contrib


def visualize_data(data):
    models = morefusion.datasets.YCBVideoModels()

    colormap = imgviz.label_colormap()
    scenes = {
        "pcd": trimesh.Scene(),
        "grid_target": trimesh.Scene(),
        "grid_nontarget_empty": trimesh.Scene(),
        "cad": trimesh.Scene(),
    }

    rgb = data["rgb"]
    depth = data["depth"]
    K = data["intrinsic_matrix"]
    pcd = morefusion.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    nonnan = ~np.isnan(depth)
    geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
    scenes["pcd"].add_geometry(geom)
    scenes["cad"].add_geometry(geom)

    T_world2cam = None
    for instance in data["instances"]:
        if T_world2cam is None:
            T_world2cam = np.linalg.inv(instance["T_cam2world"])

        class_id = instance["class_id"]
        transform = instance["transform_init"]
        grid_target = instance["grid_target"]
        grid_nontarget_empty = instance["grid_nontarget_empty_noground"]

        cad = models.get_cad(class_id=class_id)
        if hasattr(cad.visual, "to_color"):
            cad.visual = cad.visual.to_color()

        scenes["cad"].add_geometry(
            cad,
            node_name=str(instance["id"]),
            geom_name=str(instance["id"]),
            transform=transform,
        )

        transform_vg = ttf.scale_and_translate(
            scale=instance["pitch"], translate=instance["origin"]
        )

        geom = trimesh.voxel.VoxelGrid(
            grid_target, transform=transform_vg
        ).as_boxes(colors=colormap[instance["id"] + 1])
        scenes["grid_target"].add_geometry(geom)

        geom = trimesh.voxel.VoxelGrid(
            grid_nontarget_empty, transform=transform_vg
        ).as_boxes(colors=colormap[instance["id"] + 1])
        scenes["grid_nontarget_empty"].add_geometry(geom)

    camera_transform = morefusion.extra.trimesh.to_opengl_transform()
    for scene in scenes.values():
        scene.add_geometry(contrib.grid(scale=0.1), transform=T_world2cam)
        scene.camera_transform = camera_transform

    return scenes


if __name__ == "__main__":
    scenes = visualize_data(contrib.get_data())
    morefusion.extra.trimesh.display_scenes(scenes)

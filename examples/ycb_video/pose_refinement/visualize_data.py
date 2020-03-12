#!/usr/bin/env python

import functools

import imgviz
import numpy as np
import trimesh
import trimesh.transformations as ttf
import yaml

import morefusion


def grid(scale=1):
    path = []
    N = 10
    for i in range(2):
        for x in range(-N, N + 1):
            if i == 0:
                path.append(trimesh.load_path([[x, -N], [x, N]]))
            else:
                path.append(trimesh.load_path([[-N, x], [N, x]]))
    path = functools.reduce(lambda x, y: x + y, path)
    path.apply_scale(scale)
    return path


def main():
    models = morefusion.datasets.YCBVideoModels()

    colormap = imgviz.label_colormap()
    scenes = {
        "cad": trimesh.Scene(),
        "grid_target": trimesh.Scene(),
        "grid_nontarget_empty": trimesh.Scene(),
        "pcd": trimesh.Scene(),
    }
    T_world2cam = None
    for instance_id in range(3):
        instance = dict(np.load(f"data/{instance_id:08d}.npz"))

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
            node_name=str(-1),
            geom_name=str(instance["class_id"]),
            transform=transform,
        )

        transform_vg = ttf.scale_and_translate(
            scale=instance["pitch"], translate=instance["origin"]
        )

        geom = trimesh.voxel.VoxelGrid(
            grid_target, transform=transform_vg
        ).as_boxes(colors=colormap[instance_id + 1])
        scenes["grid_target"].add_geometry(geom)

        geom = trimesh.voxel.VoxelGrid(
            grid_nontarget_empty, transform=transform_vg
        ).as_boxes(colors=colormap[instance_id + 1])
        scenes["grid_nontarget_empty"].add_geometry(geom)

        rgb = imgviz.io.imread("data/image.png")
        depth = np.load("data/depth.npz")["arr_0"]
        depth = depth.astype(np.float32) / 1000
        with open(f"data/camera_info.yaml") as f:
            camera_info = yaml.safe_load(f)
        K = np.array(camera_info["K"]).reshape(3, 3)
        pcd = morefusion.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        nonnan = ~np.isnan(depth)
        geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
        scenes["pcd"].add_geometry(geom)

    camera_transform = morefusion.extra.trimesh.to_opengl_transform()
    for scene in scenes.values():
        scene.add_geometry(grid(scale=0.1), transform=T_world2cam)
        scene.camera_transform = camera_transform

    return scenes


if __name__ == "__main__":
    morefusion.extra.trimesh.display_scenes(main())

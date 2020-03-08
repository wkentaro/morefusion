#!/usr/bin/env python

import imgviz
import numpy as np
import trimesh
import trimesh.transformations as ttf

import morefusion


def get_scene(dataset):
    camera = trimesh.scene.Camera(fov=(30, 22.5))
    index = 0
    frame = dataset.get_frame(index)
    examples = dataset.get_example(index)

    scenes = {
        "rgb": None,
    }

    camera_transform = morefusion.extra.trimesh.to_opengl_transform()

    vizs = [frame["rgb"]]
    for i, example in enumerate(examples):
        viz = imgviz.tile(
            [
                example["rgb"],
                imgviz.depth2rgb(example["pcd"][:, :, 0]),
                imgviz.depth2rgb(example["pcd"][:, :, 1]),
                imgviz.depth2rgb(example["pcd"][:, :, 2]),
            ],
            border=(255, 255, 255),
        )
        viz = imgviz.draw.text_in_rectangle(
            viz,
            "lt",
            f"visibility: {example['visibility']:.0%}",
            size=30,
            background=(0, 255, 0),
            color=(0, 0, 0),
        )
        vizs.append(viz)

        geom = trimesh.voxel.VoxelGrid(
            example["grid_target"],
            ttf.scale_and_translate(example["pitch"], example["origin"]),
        ).as_boxes(colors=(1.0, 0, 0, 0.5))
        scenes[f"occupied_{i:04d}"] = trimesh.Scene(
            geom, camera=camera, camera_transform=camera_transform
        )

        geom = trimesh.voxel.VoxelGrid(
            example["grid_nontarget"],
            ttf.scale_and_translate(example["pitch"], example["origin"]),
        ).as_boxes(colors=(0, 1.0, 0, 0.5))
        scenes[f"occupied_{i:04d}"].add_geometry(geom)

        geom = trimesh.voxel.VoxelGrid(
            example["grid_empty"],
            ttf.scale_and_translate(example["pitch"], example["origin"]),
        ).as_boxes(colors=(0.5, 0.5, 0.5, 0.5))
        scenes[f"empty_{i:04d}"] = trimesh.Scene(
            geom, camera=camera, camera_transform=camera_transform
        )

        scenes[f"full_occupied_{i:04d}"] = trimesh.Scene(
            camera=camera, camera_transform=camera_transform
        )
        if (example["grid_target_full"] > 0).any():
            geom = trimesh.voxel.VoxelGrid(
                example["grid_target_full"],
                ttf.scale_and_translate(example["pitch"], example["origin"]),
            ).as_boxes(colors=(1.0, 0, 0, 0.5))
            scenes[f"full_occupied_{i:04d}"].add_geometry(geom)

        if (example["grid_nontarget_full"] > 0).any():
            colors = imgviz.label2rgb(
                example["grid_nontarget_full"].reshape(1, -1) + 1
            ).reshape(example["grid_nontarget_full"].shape + (3,))
            geom = trimesh.voxel.VoxelGrid(
                example["grid_nontarget_full"],
                ttf.scale_and_translate(example["pitch"], example["origin"]),
            ).as_boxes(colors=colors)
            scenes[f"full_occupied_{i:04d}"].add_geometry(geom)

        dim = example["grid_target"].shape[0]
        extents = np.array([dim, dim, dim]) * example["pitch"]
        geom = trimesh.path.creation.box_outline(extents)
        geom.apply_translation(
            example["origin"] + (dim / 2 - 0.5) * example["pitch"]
        )
        scenes[f"occupied_{i:04d}"].add_geometry(geom)
        scenes[f"empty_{i:04d}"].add_geometry(geom)
    viz = imgviz.tile(vizs)

    scenes["rgb"] = viz

    return scenes


if __name__ == "__main__":
    dataset = morefusion.datasets.YCBVideoRGBDPoseEstimationDataset(
        split="train"
    )
    morefusion.extra.trimesh.display_scenes(
        get_scene(dataset), height=int(320 * 0.8), width=int(480 * 0.8)
    )

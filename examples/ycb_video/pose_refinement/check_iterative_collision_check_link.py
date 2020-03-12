#!/usr/bin/env python

import chainer
from chainer.backends import cuda
import numpy as np
import tqdm

import morefusion

import contrib
from visualize_data import visualize_data


def get_scenes():
    data = contrib.get_data()

    scenes = visualize_data(data)

    models = morefusion.datasets.YCBVideoModels()

    transform = []
    points = []
    sdf = []
    pitch = []
    origin = []
    grid_target = []
    grid_nontarget_empty = []
    for instance in data["instances"]:
        transform.append(instance["transform_init"].astype(np.float32))
        points_i, sdf_i = models.get_sdf(class_id=instance["class_id"])
        points.append(cuda.to_gpu(points_i).astype(np.float32))
        sdf.append(cuda.to_gpu(sdf_i).astype(np.float32))
        pitch.append(instance["pitch"].astype(np.float32))
        origin.append(instance["origin"].astype(np.float32))
        grid_target.append(instance["grid_target"].astype(np.float32))
        grid_nontarget_empty.append(
            instance["grid_nontarget_empty"].astype(np.float32)
        )
    pitch = cuda.cupy.asarray(pitch)
    origin = cuda.cupy.asarray(origin)
    grid_target = cuda.cupy.asarray(grid_target)
    grid_nontarget_empty = cuda.cupy.asarray(grid_nontarget_empty)

    link = morefusion.contrib.IterativeCollisionCheckLink(
        transform, sdf_offset=0.01
    )
    link.to_gpu()
    optimizer = chainer.optimizers.Adam(alpha=0.01)
    optimizer.setup(link)
    link.translation.update_rule.hyperparam.alpha *= 0.1

    for i in tqdm.trange(200):
        transform = morefusion.functions.transformation_matrix(
            link.quaternion, link.translation
        )
        transform = cuda.to_cpu(transform.array)
        for j, instance in enumerate(data["instances"]):
            cad = models.get_cad(instance["class_id"])
            if hasattr(cad.visual, "to_color"):
                cad.visual = cad.visual.to_color()
            scenes["cad"].add_geometry(
                cad,
                node_name=str(instance["id"]),
                geom_name=str(instance["id"]),
                transform=transform[j],
            )
        yield scenes

        loss = link(
            points, sdf, pitch, origin, grid_target, grid_nontarget_empty
        )
        loss.backward()
        optimizer.update()
        link.zerograds()


def main():
    scenes = get_scenes()
    morefusion.extra.trimesh.display_scenes(scenes, tile=(2, 2))


if __name__ == "__main__":
    main()

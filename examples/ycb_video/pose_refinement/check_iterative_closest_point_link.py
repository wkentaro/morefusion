#!/usr/bin/env python

import chainer
from chainer.backends import cuda
import numpy as np

import morefusion

import contrib
from visualize_data import visualize_data


def get_scenes():
    data = contrib.get_data()

    scenes = visualize_data(data)

    models = morefusion.datasets.YCBVideoModels()

    links = []
    pcds_cad = []
    pcds_depth = []
    for instance in data["instances"]:
        transform = instance["transform_init"].astype(np.float32)
        link = morefusion.contrib.IterativeClosestPointLink(transform)
        link.to_gpu()
        links.append(link)

        pcd_cad = models.get_pcd(class_id=instance["class_id"])
        pcds_cad.append(cuda.to_gpu(pcd_cad.astype(np.float32)))

        pcd_depth = np.argwhere(instance["grid_target"] >= 0.5)
        pcd_depth = (
            pcd_depth.astype(np.float32) * instance["pitch"]
            + instance["origin"]
        )
        pcds_depth.append(cuda.to_gpu(pcd_depth.astype(np.float32)))

    optimizer = chainer.optimizers.Adam(alpha=0.01)
    optimizer.setup(chainer.ChainList(*links))
    for link in links:
        link.translation.update_rule.hyperparam.alpha *= 0.1

    for i in range(200):
        for j, instance in enumerate(data["instances"]):
            cad = models.get_cad(instance["class_id"])
            if hasattr(cad.visual, "to_color"):
                cad.visual = cad.visual.to_color()
            transform = cuda.to_cpu(links[j].T.array)
            scenes["cad"].add_geometry(
                cad,
                node_name=str(instance["id"]),
                geom_name=str(instance["id"]),
                transform=transform,
            )
        yield scenes

        loss = 0
        for link, pcd_cad, pcd_depth in zip(links, pcds_cad, pcds_depth):
            loss += link(pcd_cad, pcd_depth)

        loss.backward()
        optimizer.update()
        optimizer.target.zerograds()


def main():
    scenes = get_scenes()
    morefusion.extra.trimesh.display_scenes(scenes)


if __name__ == "__main__":
    main()

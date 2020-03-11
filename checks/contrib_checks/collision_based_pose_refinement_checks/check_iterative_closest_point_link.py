#!/usr/bin/env python

import argparse

import chainer
from chainer.backends import cuda
import numpy as np
import trimesh

import morefusion


def get_scenes():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_dir")
    args = parser.parse_args()

    instances = []
    for instance_id in range(3):
        instances.append(np.load(f"{args.data_dir}/{instance_id:08d}.npz"))

    models = morefusion.datasets.YCBVideoModels()

    links = []
    pcds_cad = []
    pcds_depth = []
    for instance in instances:
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

    scenes = {"scene": trimesh.Scene()}
    for i in range(200):
        for j, instance in enumerate(instances):
            cad = models.get_cad(instance["class_id"])
            if hasattr(cad.visual, "to_color"):
                cad.visual = cad.visual.to_color()
            transform = cuda.to_cpu(links[j].T.array)
            scenes["scene"].add_geometry(
                cad,
                node_name=str(j),
                geom_name=str(instance["class_id"]),
                transform=transform,
            )
        scenes[
            "scene"
        ].camera_transform = morefusion.extra.trimesh.to_opengl_transform()
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

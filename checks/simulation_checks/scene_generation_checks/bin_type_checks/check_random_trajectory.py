#!/usr/bin/env python

import imgviz
import numpy as np
import pybullet

import morefusion


def main():
    models = morefusion.datasets.YCBVideoModels()

    random_state = np.random.RandomState(0)
    generator = morefusion.simulation.BinTypeSceneGeneration(
        extents=(0.3, 0.5, 0.3),
        models=models,
        n_object=5,
        random_state=random_state,
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )
    generator.generate()

    Ts_cam2world = generator.random_camera_trajectory()

    # visualize
    def images(generator, Ts_cam2world):
        depth2rgb = imgviz.Depth2RGB()
        n_points = len(Ts_cam2world)
        for i, T_cam2world in enumerate(Ts_cam2world):
            # generator.debug_render(T_cam2world)

            rgb, depth, ins, cls = generator.render(
                T_cam2world, fovy=45, height=480, width=640,
            )
            viz = imgviz.tile(
                [
                    rgb,
                    depth2rgb(depth),
                    imgviz.label2rgb(ins + 1, rgb),
                    imgviz.label2rgb(cls, rgb),
                ],
                border=(255, 255, 255),
            )
            viz = imgviz.resize(viz, width=1000)

            font_size = 25
            text = f"{i + 1:04d} / {n_points:04d}"
            size = imgviz.draw.text_size(text, font_size)
            viz = imgviz.draw.rectangle(
                viz, (1, 1), size, outline=(0, 255, 0), fill=(0, 255, 0)
            )
            viz = imgviz.draw.text(
                viz, (1, 1), text, color=(0, 0, 0), size=font_size
            )

            imgviz.io.cv_imshow(viz)
            imgviz.io.cv_waitkey(10)

    images(generator, Ts_cam2world)


if __name__ == "__main__":
    main()

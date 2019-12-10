#!/usr/bin/env python

import imgviz
import pybullet

import morefusion


models = morefusion.datasets.YCBVideoModels()
cad_file = models.get_cad_file(class_id=2)

morefusion.extra.pybullet.init_world()

pybullet.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=45,
    cameraPitch=-45,
    cameraTargetPosition=(0, 0, 0),
)

morefusion.extra.pybullet.add_model(
    visual_file=cad_file,
    position=(0, 0, 0.3),
)
for _ in range(1000):
    pybullet.stepSimulation()

rgb, depth, segm = morefusion.extra.pybullet.get_debug_visualizer_image()

morefusion.extra.pybullet.del_world()

viz = imgviz.tile(
    [rgb, imgviz.depth2rgb(depth), imgviz.label2rgb(segm)],
    shape=(1, 3),
    border=(255, 255, 255),
)
viz = imgviz.resize(viz, width=1500)
imgviz.io.pyglet_imshow(viz)
imgviz.io.pyglet_run()

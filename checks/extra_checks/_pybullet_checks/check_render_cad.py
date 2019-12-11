#!/usr/bin/env python

import morefusion

import imgviz
import numpy as np
import trimesh


models = morefusion.datasets.YCBVideoModels()
visual_file = models.get_cad_file(class_id=8)
camera = trimesh.scene.Camera(resolution=(640, 480), fov=(60, 45))

T_cam2cad = morefusion.geometry.look_at(
    eye=(0, 0.05, 0.3),
    target=(0, 0, 0),
    up=(0, 0, -1),
)
T_cad2cam = np.linalg.inv(T_cam2cad)

# -----------------------------------------------------------------------------
# in camera coords

scene = trimesh.Scene()
cad = trimesh.load(str(visual_file))
cad.apply_transform(T_cad2cam)
scene.add_geometry(cad)
axis = trimesh.creation.axis(0.01)
scene.add_geometry(axis.copy())
axis.apply_transform(T_cad2cam)
scene.add_geometry(axis.copy())
scene.camera.resolution = camera.resolution
scene.camera.fov = camera.fov
scene.camera_transform = morefusion.extra.trimesh.to_opengl_transform()
scene.show(
    resolution=camera.resolution,
    caption='trimesh',
    start_loop=False,
)

fovy = camera.fov[1]
width, height = camera.resolution
rgb, depth, mask = morefusion.extra.pybullet.render_cad(
    visual_file, T_cad2cam, fovy, height, width
)
imgviz.io.pyglet_imshow(rgb, 'pybullet')
imgviz.io.pyglet_run()

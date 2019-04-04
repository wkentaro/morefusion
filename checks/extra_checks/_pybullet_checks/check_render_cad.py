#!/usr/bin/env python

import objslampp

import imgviz
import trimesh


models = objslampp.datasets.YCBVideoModels()
visual_file = models.get_model(class_id=8)['textured_simple']
camera = trimesh.scene.Camera(resolution=(640, 480), fov=(60, 45))

T_cad2cam = objslampp.geometry.look_at(
    eye=(0, 0.05, 0.3),
    target=(0, 0, 0),
    up=(0, -1, 0),
)

# -----------------------------------------------------------------------------

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
scene.camera.transform = objslampp.extra.trimesh.camera_transform()
scene.show(
    resolution=camera.resolution,
    caption='trimesh',
    start_loop=False,
)

fovy = camera.fov[1]
width, height = camera.resolution
rgb, depth, mask = objslampp.extra.pybullet.render_cad(
    visual_file, T_cad2cam, fovy, height, width
)
imgviz.io.pyglet_imshow(rgb, 'pybullet')
imgviz.io.pyglet_run()

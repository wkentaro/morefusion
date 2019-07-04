#!/usr/bin/env python

import trimesh

import objslampp


models = objslampp.datasets.YCBVideoModels()

voxel = models.get_solid_voxel(class_id=1)

scene = trimesh.Scene()
scene.add_geometry(voxel.as_boxes(colors=(0, 127, 0)))
scene.add_geometry(objslampp.extra.trimesh.box_outline_from_voxel(voxel))
scene.add_geometry(trimesh.creation.axis(origin_size=0.01))
scene.show(resolution=(640, 480))

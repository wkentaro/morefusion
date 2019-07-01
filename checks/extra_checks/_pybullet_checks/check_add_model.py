#!/usr/bin/env python

import time

import pybullet

import objslampp


objslampp.extra.pybullet.init_world()

models = objslampp.datasets.YCBVideoModels()
cad_file = models.get_cad_file(class_id=2)

objslampp.extra.pybullet.add_model(
    visual_file=cad_file,
    position=(0, 0, 0.3),
)
for _ in range(1000):
    pybullet.stepSimulation()

print('unique_ids:', objslampp.extra.pybullet.unique_ids)

for _ in range(3):
    time.sleep(1)

objslampp.extra.pybullet.del_world()

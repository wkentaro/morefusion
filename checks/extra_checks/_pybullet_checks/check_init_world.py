#!/usr/bin/env python

import time

import objslampp


objslampp.extra.pybullet.init_world()

for _ in range(3):
    time.sleep(1)

objslampp.extra.pybullet.del_world()

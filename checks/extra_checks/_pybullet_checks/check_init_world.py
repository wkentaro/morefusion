#!/usr/bin/env python

import time

import morefusion


morefusion.extra.pybullet.init_world()

for _ in range(3):
    time.sleep(1)

morefusion.extra.pybullet.del_world()

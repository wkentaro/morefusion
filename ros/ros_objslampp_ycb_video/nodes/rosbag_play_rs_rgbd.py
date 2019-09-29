#!/usr/bin/env python

import subprocess

import gdown

import rospy


bag_file = gdown.cached_download(
    url='https://drive.google.com/uc?id=1vnDWKX0ByFwEYssE8W0NNEF_LlL2jqlF',
    md5='5a85b26f6df5c1c5d9cd5ba838cc493f',
)
print(bag_file)

cmd = f"rosbag play {bag_file} {' '.join(rospy.myargv()[1:])}"
print(f'+ {cmd}')
subprocess.call(cmd, shell=True)

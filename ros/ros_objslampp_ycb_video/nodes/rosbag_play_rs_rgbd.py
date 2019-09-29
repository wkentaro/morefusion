#!/usr/bin/env python

import subprocess

import gdown

import rospy


bag_file = gdown.cached_download(
    url='https://drive.google.com/uc?id=1UgiBF9QLKYFkqr6DsSbozHFjdWoUnWBK',
    md5='a8f99e911d59deb235fcc1a56bef57e1',
)
print(bag_file)

cmd = f"rosbag play {bag_file} {' '.join(rospy.myargv()[1:])}"
print(f'+ {cmd}')
subprocess.call(cmd, shell=True)

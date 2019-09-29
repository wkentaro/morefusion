#!/usr/bin/env python

import argparse
import subprocess

import gdown

import rospy


bag_file = gdown.cached_download(
    url='https://drive.google.com/uc?id=1aEs_iacrwJovvR_Lf5i_g1tYp9o2-kuY',
    md5='46dc287140f00d56e79509e192eddf15',
)
print(bag_file)

cmd = f"rosbag play {bag_file} {' '.join(rospy.myargv()[1:])}"
print(f'+ {cmd}')
subprocess.call(cmd, shell=True)

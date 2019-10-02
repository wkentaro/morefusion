#!/usr/bin/env python

import subprocess
import time

import rospy


bag_file = '/home/wkentaro/Gdrive/objslampp/ros_objslampp/rs_rgbd_2019-10-02-13-56-17.bag'  # NOQA

# wait for some nodes launched
time.sleep(10)

cmd = f"rosbag play {bag_file} {' '.join(rospy.myargv()[1:])}"
print(f'+ {cmd}')
subprocess.call(cmd, shell=True)

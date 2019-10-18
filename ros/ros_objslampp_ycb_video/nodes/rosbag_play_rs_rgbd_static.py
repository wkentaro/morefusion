#!/usr/bin/env python

import subprocess
import time

# import gdown

import rospy


# bag_file = gdown.cached_download(
#     url='https://drive.google.com/uc?id=1UxUg4IozQvQNrCALXkzk23v3fBjpFCqZ',
#     md5='3625c11cd130a06557f38b8ff390882e',
# )
bag_file = '/home/wkentaro/rs_rgbd_2019-10-18-19-53-41.bag'

# wait for some nodes launched
time.sleep(10)

cmd = f"rosbag play {bag_file} {' '.join(rospy.myargv()[1:])}"
print(f'+ {cmd}')
subprocess.call(cmd, shell=True)

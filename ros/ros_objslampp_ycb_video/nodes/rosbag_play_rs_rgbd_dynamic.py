#!/usr/bin/env python

import subprocess
import time

import gdown

import rospy


bag_file = gdown.cached_download(
    url='https://drive.google.com/uc?id=1vaPyJERDNFY7W8VUBT3JvPNXHpZ6Bqun',
    md5='5bfb6eb7f80773dd2b8a6b16a93823e9',
)

# wait for some nodes launched
time.sleep(10)

cmd = f"rosbag play {bag_file} {' '.join(rospy.myargv()[1:])}"
print(f'+ {cmd}')
subprocess.call(cmd, shell=True)

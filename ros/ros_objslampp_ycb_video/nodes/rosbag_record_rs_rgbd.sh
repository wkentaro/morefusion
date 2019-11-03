#!/bin/bash -x

rosbag record -b 0 /tf /joint_states /joint_angles /tf_static_republished /camera/color/camera_info /camera/color/image_rect_color /camera/aligned_depth_to_color/camera_info /camera/aligned_depth_to_color/image_raw --output-prefix rs_rgbd $*

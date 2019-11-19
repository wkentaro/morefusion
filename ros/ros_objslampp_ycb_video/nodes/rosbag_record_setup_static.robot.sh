#!/bin/bash -x

LOG_DIR=$(date +"%Y%m%d_%H%M%S.%N")
mkdir -p $LOG_DIR

rosparam dump > $LOG_DIR/rosparam.yaml

rosbag record -b 0 \
  /tf \
  /joint_states \
  /joint_angles \
  /tf_static_republished \
  /camera/color/camera_info \
  /camera/color/image_rect_color \
  /camera/color/image_rect_color_passthrough/output \
  /camera/aligned_depth_to_color/camera_info \
  /camera/aligned_depth_to_color/image_raw \
  /camera/free_cells_vis_array \
  /camera/mask_rcnn_instance_segmentation/output/class \
  /camera/mask_rcnn_instance_segmentation/output/label_ins \
  /camera/octomap_server/output/class \
  /camera/octomap_server/output/grids \
  /camera/octomap_server/output/grids_for_render \
  /camera/octomap_server/output/grids_noentry \
  /camera/octomap_server/output/label_rendered \
  /camera/octomap_server/output/label_tracked \
  /camera/octomap_server/output/markers_bg \
  /camera/octomap_server/output/markers_fg \
  /camera/select_picking_order/output/graph \
  /camera/select_picking_order/output/poses \
  /camera/select_picking_order/output/poses_viz \
  /camera/with_occupancy/collision_based_pose_refinement/object_mapping/output/poses \
  /camera/with_occupancy/collision_based_pose_refinement/output \
  /camera/with_occupancy/object_mapping/output/poses \
  /camera/with_occupancy/singleview_3d_pose_estimation/output \
  /camera/with_occupancy/singleview_3d_pose_estimation/output/debug/rgbd \
  /move_group/monitored_planning_scene \
  -O $LOG_DIR/setup_static.robot.bag $*

cmake_minimum_required(VERSION 2.8.3)
project(ros_objslampp_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  sensor_msgs
)

add_message_files(
  FILES
  ObjectClass.msg
  ObjectClassArray.msg
  ObjectPose.msg
  ObjectPoseArray.msg
  VoxelDimensions.msg
  VoxelGrid.msg
  VoxelGridArray.msg
)

add_service_files(
  FILES
  RenderVoxelGridArray.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

cmake_minimum_required(VERSION 2.8.3)
project(ros_objslampp_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
)

add_message_files(
  FILES
  VoxelGrid.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package()

cmake_minimum_required(VERSION 2.8.3)
project(ros_objslampp_srvs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  sensor_msgs
  ros_objslampp_msgs
)

add_service_files(
  FILES
  RenderVoxelGridArray.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  ros_objslampp_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

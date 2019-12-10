cmake_minimum_required(VERSION 2.8.3)
project(morefusion_panda_ycb_video)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  jsk_recognition_msgs
  message_generation
  moveit_msgs
  octomap_ros
  pcl_ros
  roscpp
  sensor_msgs
  tf
)

find_package(PCL REQUIRED io)

find_package(octomap REQUIRED)
add_definitions(-DOCTOMAP_NODEBUGOUT)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

generate_dynamic_reconfigure_options(
  cfg/OctomapServer.cfg
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
  MoveToHome.srv
  MoveToPose.srv
  MoveToJointPosition.srv
  SetSuction.srv
  UpdateScene.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  moveit_msgs
  morefusion_panda_ycb_video
)

catkin_package(
  DEPENDS PCL OCTOMAP
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

# ---------------------------------------------------------------------

include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})

add_executable(octomap_server src/OctomapServer.cpp)
target_link_libraries(octomap_server ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})
add_dependencies(octomap_server ${PROJECT_NAME}_gencfg)

# ---------------------------------------------------------------------

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
  DIRECTORY nodes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

install(
  TARGETS octomap_server
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

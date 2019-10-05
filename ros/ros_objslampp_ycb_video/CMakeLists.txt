cmake_minimum_required(VERSION 2.8.3)
project(ros_objslampp_ycb_video)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  jsk_recognition_msgs
  octomap_server
  roscpp
)

find_package(PCL REQUIRED io)

generate_dynamic_reconfigure_options(
  cfg/OctomapServer.cfg
)

catkin_package(
  DEPENDS PCL
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

# ---------------------------------------------------------------------

include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

add_executable(octomap_server src/OctomapServer.cpp)
add_dependencies(octomap_server ${PROJECT_NAME}_gencfg)
target_link_libraries(octomap_server ${catkin_LIBRARIES} ${PCL_LIBRARIES})

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

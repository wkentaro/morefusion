cmake_minimum_required(VERSION 2.8.3)
project(ros_objslampp_panda)

add_compile_options(-std=c++11)

find_package(Eigen3 REQUIRED)

# Eigen 3.2 (Wily) only provides EIGEN3_INCLUDE_DIR, not EIGEN3_INCLUDE_DIRS
if (NOT EIGEN3_INCLUDE_DIRS)
    set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif ()

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        geometry_msgs
        moveit_msgs
        genmsg
        message_generation
        moveit_core
        moveit_visual_tools
        moveit_ros_planning
        moveit_ros_planning_interface
        pluginlib
        geometric_shapes
        pcl_ros
        pcl_conversions
        rosbag
        tf2_ros
        tf2_eigen
        tf2_geometry_msgs
        )

find_package(Boost REQUIRED system filesystem date_time thread)

add_service_files(
        FILES
        MoveToJointPosition.srv
        MoveToPose.srv
        SetSuction.srv
        UpdateScene.srv
)

generate_messages(
        DEPENDENCIES
        std_msgs
        geometry_msgs
        moveit_msgs
)

catkin_package(
        LIBRARIES ros_objslampp_panda
        interactivity_utils
        INCLUDE_DIRS
        ${THIS_PACKAGE_INCLUDE_DIRS}
        CATKIN_DEPENDS
        moveit_core
        moveit_visual_tools
        moveit_ros_planning_interface
        message_runtime
        DEPENDS
        EIGEN3
)

###########
## Build ##
###########

## Installation
install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )
install(DIRECTORY config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )

include_directories(SYSTEM ${THIS_PACKAGE_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

add_executable(pose_service src/pose_service.cpp)
target_link_libraries(pose_service ${catkin_LIBRARIES} ${Boost_LIBRARIES})
add_dependencies(pose_service ros_objslampp_panda_gencpp)
install(TARGETS pose_service DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

add_executable(update_scene_service src/update_scene_service.cpp)
target_link_libraries(update_scene_service ${catkin_LIBRARIES} ${Boost_LIBRARIES})
add_dependencies(update_scene_service ros_objslampp_panda_gencpp)
install(TARGETS update_scene_service DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

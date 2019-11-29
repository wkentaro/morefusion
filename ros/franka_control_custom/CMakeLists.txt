cmake_minimum_required(VERSION 2.8.3)
project(franka_control_custom)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
        controller_interface
        dynamic_reconfigure
        franka_hw
        geometry_msgs
        hardware_interface
        message_generation
        pluginlib
        realtime_tools
        roscpp
        rospy
        )

find_package(Eigen3 REQUIRED)
find_package(Franka 0.5.0 REQUIRED)

add_service_files(
        FILES
        SetSuction.srv
)

add_message_files(FILES
        JointTorqueComparison.msg
        )

generate_messages(DEPENDENCIES std_msgs)

generate_dynamic_reconfigure_options(
        cfg/compliance_param.cfg
        cfg/desired_mass_param.cfg
)

catkin_package(
        LIBRARIES franka_control_custom
        CATKIN_DEPENDS
        controller_interface
        dynamic_reconfigure
        franka_hw
        geometry_msgs
        hardware_interface
        message_runtime
        pluginlib
        realtime_tools
        roscpp
        DEPENDS Franka
)

add_library(franka_control_custom
        src/cartesian_velocity_controller.cpp
        src/cartesian_impedance_controller.cpp
        src/franka_state_controller.cpp
        )

add_dependencies(franka_control_custom
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        ${PROJECT_NAME}_generate_messages_cpp
        ${PROJECT_NAME}_gencpp
        ${PROJECT_NAME}_gencfg
        )

target_link_libraries(franka_control_custom PUBLIC
        ${Franka_LIBRARIES}
        ${catkin_LIBRARIES}
        )

target_include_directories(franka_control_custom SYSTEM PUBLIC
        ${Franka_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        )
target_include_directories(franka_control_custom PUBLIC
        include
        )

## Installation
install(TARGETS franka_control_custom
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )
install(DIRECTORY config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )
install(FILES franka_control_custom_plugin.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )

## Tools
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
        RESULT_VARIABLE CLANG_TOOLS
        )
if (CLANG_TOOLS)
    file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
    file(GLOB_RECURSE HEADERS
            ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
            ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
            )
    add_format_target(franka_control_custom FILES ${SOURCES} ${HEADERS})
    add_tidy_target(franka_control_custom
            FILES ${SOURCES}
            DEPENDS franka_control_custom
            )
endif ()

// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_control_custom/cartesian_velocity_controller.h>
#include <franka_control_custom/smoothing.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <Eigen/Dense>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_control_custom {

    bool CartesianVelocityController::init(hardware_interface::RobotHW *robot_hardware,
                                           ros::NodeHandle &node_handle) {

        std::string arm_id;
        if (!node_handle.getParam("arm_id", arm_id)) {
            ROS_ERROR("CartesianVelocityController: Could not get parameter arm_id");
            return false;
        }

        velocity_cartesian_interface_ =
                robot_hardware->get<franka_hw::FrankaVelocityCartesianInterface>();
        if (velocity_cartesian_interface_ == nullptr) {
            ROS_ERROR(
                    "CartesianVelocityController: Could not get Cartesian velocity interface from "
                    "hardware");
            return false;
        }
        try {
            velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
                    velocity_cartesian_interface_->getHandle(arm_id + "_robot"));
        } catch (const hardware_interface::HardwareInterfaceException &e) {
            ROS_ERROR_STREAM(
                    "CartesianVelocityController: Exception getting Cartesian handle: " << e.what());
            return false;
        }

        auto* state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
        if (state_interface == nullptr) {
            ROS_ERROR_STREAM(
                    "CartesianImpedanceController: Error getting state interface from hardware");
            return false;
        }
        try {
            state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
                    state_interface->getHandle(arm_id + "_robot"));
        } catch (hardware_interface::HardwareInterfaceException& ex) {
            ROS_ERROR_STREAM(
                    "CartesianImpedanceController: Exception getting state handle from interface: "
                            << ex.what());
            return false;
        }

        ef_vel_subsriber = node_handle.subscribe<std_msgs::Float64MultiArray>("/end_effector_velocity", 1,
                                                                &CartesianVelocityController::commandCB, this);
        return true;
    }

    void CartesianVelocityController::starting(const ros::Time & /* time */) {}

    void CartesianVelocityController::stopping(const ros::Time & /*time*/) {}

    void CartesianVelocityController::update(const ros::Time & time,
                                             const ros::Duration & period) {

        // state
        double delta_t = period.toSec();
        franka::RobotState robot_state = state_handle_->getRobotState();

        // read target velocity from topic
        std::vector<double> target_velocity_std_vector = *command_buffer_.readFromRT();

        // if not empty
        if (!target_velocity_std_vector.empty())
        {
            // clip velocity
            Eigen::Vector3d target_velocity(target_velocity_std_vector.data());
            Eigen::Vector3d velocity_command = smooth_velocity(&robot_state, target_velocity, delta_t);

            // execute
            std::array<double, 6> command_array = {{velocity_command[0], velocity_command[1], velocity_command[2],
                                              0.0, 0.0, 0.0}};
            velocity_cartesian_handle_->setCommand(command_array);
        }
    }

    void CartesianVelocityController::commandCB(const std_msgs::Float64MultiArrayConstPtr& msg) {
        command_buffer_.writeFromNonRT(msg->data);
    }


}  // namespace franka_control_custom

PLUGINLIB_EXPORT_CLASS(franka_control_custom::CartesianVelocityController,
                       controller_interface::ControllerBase)
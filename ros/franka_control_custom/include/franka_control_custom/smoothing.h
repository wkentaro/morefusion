// Author: Enrico Corvaglia
// https://github.com/CentroEPiaggio/kuka-lwr/blob/master/lwr_controllers/include/utils/pseudo_inversion.h
// File provided under public domain
// pseudo_inverse() computes the pseudo inverse of matrix M_ using SVD decomposition (can choose
// between damped and not)
// returns the pseudo inverted matrix M_pinv_

#pragma once

#include <Eigen/Dense>

const double max_velocity = 1.7; // 1.7
const double max_acceleration = 1.0; // 13
const double min_divisor = 1e-9;

inline Eigen::Vector3d clip_velocity(Eigen::Vector3d *target_velocity_vector, franka::RobotState *robot_state,
                                     double delta_t)
{
    double target_velocity_norm = target_velocity_vector->norm();
    double clipped_target_velocity_norm = std::min(target_velocity_norm, max_velocity);
    Eigen::Vector3d clipped_target_velocity_vector;
    if (clipped_target_velocity_norm < min_divisor)
    {
        clipped_target_velocity_vector << 0, 0, 0;
    }
    else
    {
        clipped_target_velocity_vector = *target_velocity_vector/target_velocity_norm * clipped_target_velocity_norm;
    }

    // compute previous velocity command
    std::array<double,6> commanded_twist = robot_state->O_dP_EE_c;
    Eigen::Vector3d previous_velocity_command_vector;
    previous_velocity_command_vector << commanded_twist[0], commanded_twist[1], commanded_twist[2];

    // compute the acceleration for new clipped velocity
    Eigen::Vector3d target_acceleration_vector = (clipped_target_velocity_vector - previous_velocity_command_vector)/(delta_t + min_divisor);

    // clip the acceleration
    double target_acceleration_norm = target_acceleration_vector.norm();
    double clipped_acceleration_norm = std::min(target_acceleration_norm, max_acceleration);
    Eigen::Vector3d clipped_target_acceleration_vector;
    if (clipped_acceleration_norm < min_divisor)
    {
        clipped_target_acceleration_vector << 0, 0, 0;
    }
    else
    {
        clipped_target_acceleration_vector = target_acceleration_vector/target_acceleration_norm * clipped_acceleration_norm;
    }

    // return resultant clipped velocity
    return previous_velocity_command_vector + clipped_target_acceleration_vector*delta_t;
}

inline Eigen::Vector3d smooth_velocity(franka::RobotState *robot_state,
        Eigen::Vector3d target_velocity_vector, double delta_t) {

    return clip_velocity(&target_velocity_vector, robot_state, delta_t);
}

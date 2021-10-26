#include <pluginlib/class_loader.h>
#include <ros/ros.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/PlanningScene.h>
#include <moveit_msgs/GetPlanningScene.h>

#include <boost/scoped_ptr.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <chrono>

#include "ros/ros.h"
#include "morefusion_ros_panda/MoveToPose.h"

class PoseService {

    ros::NodeHandle node_handle;
    ros::ServiceClient scene_getter;
    const std::string PLANNING_GROUP = "panda_arm";
    moveit::planning_interface::MoveGroupInterface move_group;
    robot_model_loader::RobotModelLoader robot_model_loader;
    robot_model::RobotModelPtr robot_model;
    planning_scene::PlanningScenePtr planning_scene;
    planning_pipeline::PlanningPipelinePtr planning_pipeline;
    trajectory_processing::IterativeParabolicTimeParameterization iptp;
    robot_state::RobotStatePtr robot_state;
    std::vector<double> joint_values;
    const robot_state::JointModelGroup *joint_model_group;
    boost::scoped_ptr<pluginlib::ClassLoader<planning_interface::PlannerManager>> planner_plugin_loader;
    std::string planner_plugin_name;

public:
    PoseService();

    int run();

    bool move_to_pose(morefusion_ros_panda::MoveToPose::Request &service_req,
                               morefusion_ros_panda::MoveToPose::Response &service_res);
};

PoseService::PoseService() :
        node_handle("~"),
        move_group(PLANNING_GROUP),
        robot_model_loader("robot_description") {
    scene_getter = node_handle.serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");
    scene_getter.waitForExistence();
    robot_model = robot_model_loader.getModel();
    planning_scene = planning_scene::PlanningScenePtr(new planning_scene::PlanningScene(robot_model));
    planning_pipeline = planning_pipeline::PlanningPipelinePtr
            (new planning_pipeline::PlanningPipeline(robot_model, node_handle, "planning_plugin", "request_adapters"));
}

int PoseService::run() {
    ros::ServiceServer service = node_handle.advertiseService("move_to_pose", &PoseService::move_to_pose, this);
    ros::AsyncSpinner spinner(1);
    spinner.start();

    while (ros::ok()) {
        ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(0.1));
    }

    return 0;
}

bool PoseService::move_to_pose(morefusion_ros_panda::MoveToPose::Request &service_req,
                                        morefusion_ros_panda::MoveToPose::Response &service_res) {


    // load planning scene from parameter server
    moveit_msgs::GetPlanningScene srv;
    srv.request.components.components = 1023;
    scene_getter.call(srv);
    moveit_msgs::PlanningScene received_scene_msg = srv.response.scene;

    // update local planning scene from the loaded one
    moveit_msgs::PlanningScene scene_msg;
    planning_scene->getPlanningSceneMsg(scene_msg);
    scene_msg.world.collision_objects = received_scene_msg.world.collision_objects;
    scene_msg.robot_state.attached_collision_objects = received_scene_msg.robot_state.attached_collision_objects;
    scene_msg.is_diff = true;
    planning_scene->setPlanningSceneMsg(scene_msg);

    // Create a RobotState and JointModelGroup to keep track of the current robot pose and planning group
    robot_state = move_group.getCurrentState();

    // model group
    joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);

    // joint state values
    robot_state->copyJointGroupPositions(joint_model_group, joint_values);

    // Configure a valid robot state
    planning_scene->getCurrentStateNonConst().setJointGroupPositions(joint_model_group, joint_values);


    // We will get the name of planning plugin we want to load
    if (!node_handle.getParam("planning_plugin", planner_plugin_name))
        ROS_FATAL_STREAM("Could not find planner plugin name");
    try {
        planner_plugin_loader.reset(new pluginlib::ClassLoader<planning_interface::PlannerManager>(
                "moveit_core", "planning_interface::PlannerManager"));
    }
    catch (pluginlib::PluginlibException &ex) {
        ROS_FATAL_STREAM("Exception while creating planning plugin loader " << ex.what());
    }

    // Pose Goal
    planning_interface::MotionPlanRequest req;
    move_group.constructMotionPlanRequest(req);
    req.group_name = PLANNING_GROUP;
    planning_interface::MotionPlanResponse res;

    std::vector<geometry_msgs::PoseStamped> goal_poses;
    std::vector<geometry_msgs::Vector3> position_constraints = service_req.position_constraints;
    std::vector<geometry_msgs::Vector3> orientation_constraints = service_req.orientation_constraints;


    for (int i = 0; i < service_req.goal_poses.size(); i++) {

        geometry_msgs::PoseStamped goal_pose;

        goal_pose.header.frame_id = "panda_link0";
        goal_pose.pose = service_req.goal_poses[i];

        if (service_req.pure_translation)
        {
            goal_pose.pose.orientation = move_group.getCurrentPose().pose.orientation;
        }
        if (service_req.pure_rotation)
        {
            goal_pose.pose.position = move_group.getCurrentPose().pose.position;
        }

        // tolerances
        std::vector<double> tolerance_pose;
        tolerance_pose.push_back(position_constraints[i].x);
        tolerance_pose.push_back(position_constraints[i].y);
        tolerance_pose.push_back(position_constraints[i].z);

        std::vector<double> tolerance_angle;
        tolerance_angle.push_back(orientation_constraints[i].x);
        tolerance_angle.push_back(orientation_constraints[i].y);
        tolerance_angle.push_back(orientation_constraints[i].z);

        moveit_msgs::Constraints pose_goal =
                kinematic_constraints::constructGoalConstraints(service_req.link_name, goal_pose, tolerance_pose,
                                                                tolerance_angle);

        req.goal_constraints.push_back(pose_goal);
        goal_poses.push_back(goal_pose);
    }

    // We now generate plan using pipeline
    planning_pipeline->generatePlan(planning_scene, req, res);
    if (res.error_code_.val != res.error_code_.SUCCESS) {
        service_res.success = false;
        service_res.pose_reached = move_group.getCurrentPose().pose;
        return false;
    }

    // perform time parameterization
    iptp.computeTimeStamps(*res.trajectory_, service_req.velocity_scaling, service_req.acceleration_scaling);

    moveit_msgs::MotionPlanResponse response;
    res.getMessage(response);

    // Execute the motion on real robot
    moveit::planning_interface::MoveGroupInterface::Plan myplan;
    myplan.trajectory_ = response.trajectory;
    move_group.execute(myplan);

    service_res.success = true;
    service_res.pose_reached = move_group.getCurrentPose().pose;

    return true;
}

int main(int argc, char **argv) {
    const std::string node_name = "move_to_pose_server";
    ros::init(argc, argv, node_name);
    PoseService ps;
    int val = ps.run();
    return val;
}
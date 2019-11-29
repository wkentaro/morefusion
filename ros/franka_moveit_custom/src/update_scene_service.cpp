#include <unistd.h>
#include <ros/ros.h>
#include <moveit_msgs/ApplyPlanningScene.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <franka_moveit_custom/UpdateScene.h>

class SceneService{
    ros::NodeHandle node_handle;
    ros::ServiceClient planning_scene_diff_client;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    const std::string PLANNING_GROUP = "panda_arm";
    moveit::planning_interface::MoveGroupInterface move_group;
public:
    SceneService();
    int run();
    bool update_scene(franka_moveit_custom::UpdateScene::Request &service_req,
                      franka_moveit_custom::UpdateScene::Response &service_res);
};

SceneService::SceneService() :
move_group(PLANNING_GROUP)
{}

bool SceneService::update_scene(franka_moveit_custom::UpdateScene::Request &service_req,
                  franka_moveit_custom::UpdateScene::Response &service_res)
{
    // planning scene message
    moveit_msgs::PlanningScene planning_scene;

    // add world objects
    std::vector<moveit_msgs::CollisionObject> collision_objects = service_req.collision_objects;
    planning_scene.world.collision_objects.clear();
    for (int i=0; i<collision_objects.size(); i++){
        moveit_msgs::CollisionObject collision_object = collision_objects[i];
        collision_object.header.frame_id = move_group.getPlanningFrame();
        planning_scene.world.collision_objects.push_back(collision_object);
    }

    planning_scene_interface.addCollisionObjects(collision_objects);

    // process attached objects
    std::vector<moveit_msgs::AttachedCollisionObject> attached_objects = service_req.attached_objects;
    std::vector<moveit_msgs::CollisionObject> attached_collision_objects;
    for (int i=0; i<attached_objects.size(); i++){
        if (attached_objects[i].object.operation == attached_objects[i].object.REMOVE)
        {
            move_group.detachObject(attached_objects[i].object.id);
            usleep(1000000); // required in order to ensure it is removed from scene
        }
        moveit_msgs::AttachedCollisionObject attached_object = attached_objects[i];
        moveit_msgs::CollisionObject attached_collision_object = attached_object.object;
        attached_collision_objects.push_back(attached_collision_object);
    }

    planning_scene_interface.addCollisionObjects(attached_collision_objects);

    // add attached objects
    planning_scene.robot_state.attached_collision_objects.clear();
    for (int i=0; i<attached_objects.size(); i++){
        moveit_msgs::AttachedCollisionObject attached_object = attached_objects[i];
        if (attached_object.object.operation == attached_object.object.ADD)
        {
            move_group.attachObject(attached_object.object.id, attached_object.link_name);
        }
        planning_scene.robot_state.attached_collision_objects.push_back(attached_object);
    }

    planning_scene.robot_state.is_diff = true;
    planning_scene.is_diff = true;

    moveit_msgs::ApplyPlanningScene srv;
    srv.request.scene = planning_scene;
    planning_scene_diff_client.call(srv);

    service_res.success = true;

    ROS_INFO("Scene updated.");
    return true;
}

int SceneService::run()
{
    ros::ServiceServer service = node_handle.advertiseService("update_scene", &SceneService::update_scene, this);
    node_handle.serviceClient<moveit_msgs::ApplyPlanningScene>("apply_planning_scene");
    planning_scene_diff_client.waitForExistence();

    ros::spin();
}

int main(int argc, char **argv){
    const std::string node_name = "update_scene_server";
    ros::init(argc, argv, node_name);
    SceneService ss;
    int val = ss.run();
    return val;
}

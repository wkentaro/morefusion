import rospy
import robot_demo.robot_interface as robot_interface
import robot_demo.world_interface as world_interface
import objslampp
import objslampp.datasets.ycb_video as ycb_video_dataset
from geometry_msgs.msg import Pose
import trimesh

if __name__ == '__main__':

    rospy.init_node('collision_test')
    object_models = ycb_video_dataset.YCBVideoModels()
    object_filepath = object_models.get_cad_file(1)
    collision_filepath = objslampp.utils.get_collision_file(object_filepath)
    mesh_multi = trimesh.load_mesh(str(collision_filepath), process=False)
    if type(mesh_multi) is list:
        mesh = mesh_multi[0]
        for i in range(1, len(mesh_multi)):
            mesh += mesh_multi[i]
    else:
        mesh = mesh_multi

    # start robot in home
    robot_interface.move_to_home()

    # assert that it cannot go through table
    pose = Pose()
    pose.position.x = 0.77
    pose.position.y = 0.
    pose.position.z = 0.
    pose.orientation.x = 1.
    pose.orientation.y = 0.
    pose.orientation.z = 0.
    pose.orientation.w = 0.
    assert(robot_interface.set_end_effector_quaternion_pose([pose], must_succeed=False)[0] is False)
    assert(robot_interface.set_end_effector_quaternion_pointing_pose(pose, must_succeed=False)[0] is False)

    # add static object below robot
    pose = Pose()
    pose.position.x = 0.77
    pose.position.y = 0.
    pose.position.z = 0.3
    pose.orientation.x = 1.
    pose.orientation.y = 0.
    pose.orientation.z = 0.
    pose.orientation.w = 0.
    #world_interface.add_static_meshes(['obstacle'], [mesh], [pose])

    # assert that it cannot go through object
    #assert(robot_interface.set_end_effector_quaternion_pose([pose], must_succeed=False)[0] is False)
    #assert(robot_interface.set_end_effector_quaternion_pointing_pose(pose, must_succeed=False)[0] is False)

    # remove static object
    #world_interface.remove_static_meshes(['obstacle'])

    # attach the object to end effector
    #world_interface.add_attached_meshes(['attached_obstacle'], [mesh], [pose], ['panda_suction_cup'])

    # assert the robot cannot touch this into itself
    pose = Pose()
    pose.position.x = 0.0987
    pose.position.y = 0.0067
    pose.position.z = 0.461
    pose.orientation.x = 0.978
    pose.orientation.y = 0.025
    pose.orientation.z = -0.2047
    pose.orientation.w = -0.0407
    assert(robot_interface.set_end_effector_quaternion_pose([pose], must_succeed=False)[0] is False)
    assert(robot_interface.set_end_effector_quaternion_pointing_pose(pose, must_succeed=False)[0] is False)

    print('Test Completed.')

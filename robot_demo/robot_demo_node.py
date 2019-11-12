# general
import math
import time
import trimesh
import numpy as np
from threading import Lock

# ros
import tf
import rospy
from ros_objslampp_msgs.msg import ObjectPoseArray
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

# objslampp
import objslampp
import objslampp.datasets.ycb_video as ycb_video_dataset

# robot demo
import robot_demo.general_kinematics as gk
import robot_demo.world_interface as world_interface
import robot_demo.robot_interface as robot_interface
from robot_demo.object_pose_interface import ObjectPoseInterface


class RobotDemo:

    def __init__(self):

        rospy.init_node('robot_demo')
        self._tf_listener = tf.TransformListener(cache_time=rospy.Duration(1000))
        self._tf_broadcaster = tf.TransformBroadcaster()

        print('tf setup')

        self._object_models = ycb_video_dataset.YCBVideoModels()
        self._object_filepaths = [self._object_models.get_cad_file(i) for i in range(1, len(self._object_models.class_names))]
        self._collision_filepaths = [objslampp.utils.get_collision_file(cad_filename) for cad_filename in self._object_filepaths]
        self._collision_meshes_multi = [trimesh.load_mesh(str(filepath), process=False) for filepath in self._collision_filepaths]
        self._collision_meshes = list()
        for mesh_multi in self._collision_meshes_multi:
            if type(mesh_multi) is list:
                mesh = mesh_multi[0]
                for i in range(1,len(mesh_multi)):
                    mesh += mesh_multi[i]
            else:
                mesh = mesh_multi
            self._collision_meshes.append(mesh)

        self._object_pose_interface = ObjectPoseInterface(self._object_models, ['x-','x+','z-','z+','y-','y+'])

        self._robot_interface = robot_interface
        self._world_interface = world_interface

        self._all_objects_removed = False
        self._all_distractors_removed = False

        self._define_robot_poses()

        self._object_id_to_grasp = None
        self._picked_objects = list()

        self._grasp_overlap = 0.0065
        self._pre_placement_z_dist = 0.005

        self._over_target_box_pose = Pose()
        self._over_target_box_pose.position = Point(0.4, -0.45, 0.575)
        self._over_target_box_pose.orientation = Quaternion(0.8, -0.6, 0.008, -0.01)

        self._in_target_box_position = [0.4, -0.45]

        self._over_distractor_box_pose = Pose()
        self._over_distractor_box_pose.position = Point(0.485, 0.445, 0.585)
        self._over_distractor_box_pose.orientation = Quaternion(0.891, 0.45, 0.034, -0.0192)

        self._in_distractor_box_pose = Pose()
        self._in_distractor_box_pose.position = Point(0.485, 0.445, 0.372)
        self._in_distractor_box_pose.orientation = Quaternion(0.891, 0.45, 0.034, -0.0192)

        self._pick_point_np_poses_in_world_frame = dict()
        self._pick_point_mats_in_world_frame = dict()

        self._object_pose_msgs_in_world_frame = dict()
        self._object_mats_in_world_frame = dict()

        self._lock = Lock()

        # planning scene #
        self._static_mesh_ids = list()

    # Object Pose Functions #
    # ----------------------#

    def _broadcast_object_pose(self):
        translation = self._pick_point_np_poses_in_world_frame[self._object_id_to_grasp][0:3]
        rotation = self._pick_point_np_poses_in_world_frame[self._object_id_to_grasp][3:]
        self._tf_broadcaster.sendTransform(translation, rotation, self._object_ros_time, 'object', 'panda_link0')

    def _get_grasp_pose(self):

        grasp_mat_in_obj_frame = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, self._grasp_overlap],
                                           [0, 0, 0, 1]])

        grasp_mat_in_world_frame = np.matmul(self._pick_point_mats_in_world_frame[self._object_id_to_grasp], grasp_mat_in_obj_frame)
        self._grasp_pose_in_world_frame = gk.mat_to_quaternion_pose(np, grasp_mat_in_world_frame)
        translation = self._grasp_pose_in_world_frame[0:3]
        rotation = self._grasp_pose_in_world_frame[3:]

        self._tf_broadcaster.sendTransform(translation, rotation, self._object_ros_time, 'grasp', 'panda_link0')

        pose = Pose()
        pose.position = Point(translation[0], translation[1], translation[2])
        pose.orientation = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])

        return pose

    def _get_pre_grasp_pose(self):

        pre_grasp_mat_in_obj_frame = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, -0.05],
                                               [0, 0, 0, 1]])

        pre_grasp_mat_in_world_frame = np.matmul(self._pick_point_mats_in_world_frame[self._object_id_to_grasp], pre_grasp_mat_in_obj_frame)
        pre_grasp_pose_in_world_frame = gk.mat_to_quaternion_pose(np, pre_grasp_mat_in_world_frame)
        translation = pre_grasp_pose_in_world_frame[0:3]
        rotation = pre_grasp_pose_in_world_frame[3:]

        self._tf_broadcaster.sendTransform(translation, rotation, self._object_ros_time, 'pre_grasp', 'panda_link0')

        pose = Pose()
        pose.position = Point(translation[0], translation[1], translation[2])
        pose.orientation = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])

        return pose

    # Robot Functions #
    # ----------------#

    def _define_robot_poses(self):

        _home_position = np.array([0.7, 0, 0.6])
        _q1 = gk.quaternion_from_vector_and_angle(np, [1, 0, 0], math.pi)
        _q2 = gk.quaternion_from_vector_and_angle(np, [0, 0, 1], -math.pi / 4)
        _home_quaternion = gk.hamilton_product(np, _q1, _q2)
        home_pose = np.concatenate((_home_position, _home_quaternion), -1)

        x_offset = 0.2

        robot_position_offsets = [np.array([x_offset, 0, 0]),
                                  np.array([x_offset, 0, -0.1])]

        robot_rotation_vectors = [np.array([0, 0, 0]),
                                  np.array([0, 0, 0])]

        robot_quaternion_offsets = [gk.rotation_vector_to_quaternion(np, aa) for aa in robot_rotation_vectors]
        robot_positions = [home_pose[0:3] + offset for offset in robot_position_offsets]
        robot_quaternions = [gk.hamilton_product(np, home_pose[3:], qt) for qt in robot_quaternion_offsets]
        self._robot_poses = [np.concatenate((pos, quat),-1) for pos, quat in zip(robot_positions, robot_quaternions)]

    def _initialization_motion(self):
        self._robot_interface.move_to_home(0.025, 0.025)
        for robot_pose in self._robot_poses:

            pose = Pose()
            pose.position = Point(robot_pose[0], robot_pose[1], robot_pose[2])
            pose.orientation = Quaternion(robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6])

            self._robot_interface.set_end_effector_quaternion_pose_linearly(pose, 0.025, 0.025)
        self._robot_interface.move_to_home(0.025, 0.025)

    def _move_robot_over_table(self):
        self._robot_interface.move_to_home(0.7, 0.7)

    def _move_robot_to_pre_grasp_pose(self, pre_grasp_pose):
        self._robot_interface.set_end_effector_quaternion_pointing_pose(pre_grasp_pose)

    def _move_robot_to_grasp_pose(self, grasp_pose):
        _, pose_reached = self._robot_interface.set_end_effector_position_linearly(grasp_pose.position, 0.25, 0.25)
        return pose_reached

    def _move_robot_to_post_grasp_pose(self, post_grasp_pose):
        self._robot_interface.set_end_effector_position_linearly(post_grasp_pose.position, 0.25, 0.25)

    def _suction_grip_object(self):
        self._robot_interface.set_suction_state(True)
        time.sleep(1)

    def _move_robot_over_target_box(self):
        self._robot_interface.set_end_effector_quaternion_pose_linearly(self._over_target_box_pose, 0.7, 0.7)

    def _move_robot_over_distractor_box(self):
        self._robot_interface.set_end_effector_quaternion_pose_linearly(self._over_distractor_box_pose, 0.7, 0.7)

    def _move_robot_to_pre_place_pose(self, robot_poses):
        robot_pre_poses = list()
        for robot_pose in robot_poses:
            robot_pre_pose = robot_pose
            robot_pre_pose.position.z += self._pre_placement_z_dist
            robot_pre_poses.append(robot_pre_pose)
        _, pre_pose_reached = self._robot_interface.set_end_effector_quaternion_pose(robot_pre_poses, 0.7, 0.7)
        return pre_pose_reached

    def _move_robot_to_place_pose(self, pre_pose_reached, object_to_robot_mat):
        place_position = pre_pose_reached.position
        place_position.z -= self._pre_placement_z_dist
        _, robot_pose = self._robot_interface.set_end_effector_position_linearly(place_position, 0.7, 0.7)
        pos = robot_pose.position
        ori = robot_pose.orientation
        robot_np_pose = np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        object_mat = np.matmul(gk.quaternion_pose_to_mat(np, robot_np_pose), np.linalg.inv(object_to_robot_mat))
        object_np_pose = gk.mat_to_quaternion_pose(np, object_mat)
        object_pose = Pose()
        object_pose.position.x = object_np_pose[0]
        object_pose.position.y = object_np_pose[1]
        object_pose.position.z = object_np_pose[2]
        object_pose.orientation.x = object_np_pose[3]
        object_pose.orientation.y = object_np_pose[4]
        object_pose.orientation.z = object_np_pose[5]
        object_pose.orientation.w = object_np_pose[6]
        return object_pose

    def _move_robot_to_drop_pose(self):
        self._robot_interface.set_end_effector_position_linearly(self._in_distractor_box_pose.position, 0.7, 0.7)
        return self._in_distractor_box_pose

    def _release_suction_grip(self):
        self._robot_interface.set_suction_state(False)
        time.sleep(8)

    # Object Checking #
    # ----------------#

    def _check_if_all_objects_removed(self):
        if len(self._picked_objects) < len(self._ordered_object_ids_to_grasp):
            return False
        return True

    def _check_if_all_distractors_removed(self):
        if len(self._picked_objects) < len(self._ordered_object_ids_to_grasp) - 1:
            return False
        return True

    def _choose_next_object_to_grasp(self):
        for id in self._ordered_object_ids_to_grasp:
            if id in self._picked_objects:
                continue
            return id

    # Object Picking Poses #
    # ---------------------#

    def _pick_poses_callback(self, pick_point_poses):

        if len(pick_point_poses.poses) == 0:
            return

        self._object_ros_time = pick_point_poses.header.stamp
        frame_id = pick_point_poses.header.frame_id

        for object_pose in pick_point_poses.poses:

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = frame_id
            pose_stamped.header.stamp = self._object_ros_time
            pose_stamped.pose = object_pose.pose

            pick_point_pose_in_world_frame = self._tf_listener.transformPose('panda_link0', pose_stamped).pose

            pos = pick_point_pose_in_world_frame.position
            ori = pick_point_pose_in_world_frame.orientation
            pick_point_np_pose_in_world_frame = np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
            self._pick_point_np_poses_in_world_frame[object_pose.class_id] = pick_point_np_pose_in_world_frame

            pick_point_mat_in_world_frame = gk.quaternion_pose_to_mat(np, pick_point_np_pose_in_world_frame)
            self._pick_point_mats_in_world_frame[object_pose.class_id] = pick_point_mat_in_world_frame

        self._ordered_object_ids_to_grasp = [object_pose.class_id for object_pose in pick_point_poses.poses]

        print('object tree:')
        for class_id in self._ordered_object_ids_to_grasp:
            print(ycb_video_dataset.class_names[class_id])

    # object Poses #
    # -------------#

    def _object_poses_callback(self, object_poses):

        if len(object_poses.poses) == 0:
            return

        self._object_ros_time = object_poses.header.stamp
        frame_id = object_poses.header.frame_id

        for object_pose in object_poses.poses:

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = frame_id
            pose_stamped.header.stamp = self._object_ros_time
            pose_stamped.pose = object_pose.pose

            object_pose_in_world_frame = self._tf_listener.transformPose('panda_link0', pose_stamped).pose

            pos = object_pose_in_world_frame.position
            ori = object_pose_in_world_frame.orientation
            object_np_pose_in_world_frame = np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])

            object_mat_in_world_frame = gk.quaternion_pose_to_mat(np, object_np_pose_in_world_frame)
            self._object_mats_in_world_frame[object_pose.class_id] = object_mat_in_world_frame

            self._object_pose_msgs_in_world_frame[object_pose.class_id] = object_pose_in_world_frame


    # Scene Updating #
    # ---------------#

    def _update_static_scene(self):

        self._world_interface.remove_attached_meshes(self._static_mesh_ids, ['panda_link0']*len(self._static_mesh_ids))
        self._static_mesh_ids.clear()

        mesh_ids = list()
        meshes = list()
        poses = list()

        for class_id, object_pose in self._object_pose_msgs_in_world_frame.items():
            if class_id in self._picked_objects:
                continue
            mesh_ids.append(str(class_id))
            self._static_mesh_ids.append(str(class_id))
            poses.append(object_pose)
            meshes.append(self._collision_meshes[class_id-1])

        self._world_interface.add_attached_meshes(mesh_ids, meshes, poses, ['panda_link0']*len(mesh_ids))

    def _update_scene_with_grasp(self):
        self._world_interface.remove_attached_meshes([str(self._object_id_to_grasp)], ['panda_link0'])
        mesh_to_grasp = self._collision_meshes[self._object_id_to_grasp - 1]
        mesh_pose = self._object_pose_msgs_in_world_frame[self._object_id_to_grasp]
        self._world_interface.add_attached_meshes([str(self._object_id_to_grasp)], [mesh_to_grasp], [mesh_pose], ['panda_suction_cup'])

    def _update_scene_with_drop(self):
        self._world_interface.remove_attached_meshes([str(self._object_id_to_grasp)], ['panda_suction_cup'])

    def _update_scene_with_placement(self, pose):
        self._world_interface.remove_attached_meshes([str(self._object_id_to_grasp)], ['panda_suction_cup'])
        self._world_interface.add_attached_meshes([str(self._object_id_to_grasp)],
                                                [self._collision_meshes[self._object_id_to_grasp - 1]], [pose], ['panda_link0'])

    # Run #
    # ----#

    def run(self):

        # scene inference #
        # ----------------#

        print('performing initialization motion...')
        self._initialization_motion()

        print('waiting for object tree')
        try:
            object_picking_poses = rospy.wait_for_message('/camera/select_picking_order/output/poses', ObjectPoseArray, timeout=30)
        except:
            raise Exception('Object Tree Not Found.')
        self._pick_poses_callback(object_picking_poses)

        try:
            object_poses = rospy.wait_for_message('/camera/with_occupancy/collision_based_pose_refinement/object_mapping/output/poses', ObjectPoseArray, timeout=30)
        except:
            raise Exception('Object Poses Not Found.')
        self._object_poses_callback(object_poses)
        self._update_static_scene()

        print('initialization complete')

        while not self._all_objects_removed:

            # object selection #
            # -----------------#

            self._object_id_to_grasp = self._choose_next_object_to_grasp()
            print('picking up ' + str(ycb_video_dataset.class_names[self._object_id_to_grasp]))

            self._broadcast_object_pose()

            # pose calculation #
            # -----------------#

            pre_grasp_pose = self._get_pre_grasp_pose()
            grasp_pose = self._get_grasp_pose()
            inv_object_mat = np.linalg.inv(self._object_mats_in_world_frame[self._object_id_to_grasp])
            post_grasp_pose = pre_grasp_pose

            # Distractors #
            # ------------#

            self._all_distractors_removed = self._check_if_all_distractors_removed()

            # move to object #
            # ---------------#

            self._move_robot_over_table()
            self._move_robot_to_pre_grasp_pose(pre_grasp_pose)
            robot_pose = self._move_robot_to_grasp_pose(grasp_pose)

            # update robot pose #
            # ------------------#

            pos = robot_pose.position
            ori = robot_pose.orientation
            robot_np_pose = np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
            object_to_robot_mat = np.matmul(inv_object_mat, gk.quaternion_pose_to_mat(np, robot_np_pose))

            # move object #
            # ------------#

            self._suction_grip_object()

            self._update_scene_with_grasp()

            self._move_robot_to_post_grasp_pose(post_grasp_pose)

            self._move_robot_over_table()
            if self._all_distractors_removed:
                self._move_robot_over_target_box()

                # get possible robot poses
                robot_poses = self._object_pose_interface.get_robot_poses(self._object_id_to_grasp, object_to_robot_mat,
                                                                          self._in_target_box_position)
                # make motion
                robot_pose = self._move_robot_to_pre_place_pose(robot_poses)
                obj_pose = self._move_robot_to_place_pose(robot_pose, object_to_robot_mat)
                self._update_scene_with_placement(obj_pose)
            else:
                self._move_robot_over_distractor_box()
                self._move_robot_to_drop_pose()
                self._update_scene_with_drop()
            self._release_suction_grip()

            # object logging #
            # ---------------#

            self._picked_objects.append(self._object_id_to_grasp)
            self._all_objects_removed = self._check_if_all_objects_removed()

            # reset robot #
            # ------------#

            self._move_robot_over_table()
            print('pick completed')

        print('Demo completed!')


def main():
    robot_demo = RobotDemo()
    input('press enter to continue demo')
    robot_demo.run()


if __name__ == '__main__':
    main()

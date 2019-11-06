import tf
import time
import math
import numpy as np
import robot_demo.general_kinematics as gk
import rospy
from robot_demo.robot_interface import RobotInterface
from ros_objslampp_msgs.msg import ObjectPoseArray
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
import objslampp.datasets.ycb_video as ycb_video_dataset
from threading import Lock

class RobotDemo:

    def __init__(self):

        self._object_models = ycb_video_dataset.YCBVideoModels()

        rospy.init_node('robot_demo')
        self._robot_interface = RobotInterface()

        self._all_objects_removed = False
        self._all_distractors_removed = False

        self._define_robot_poses()

        self._object_id_to_grasp = None
        self._picked_objects = list()

        self._grasp_overlap = 0.0065

        self._over_target_box_pose = Pose()
        self._over_target_box_pose.position = Point(0.386, -0.492, 0.575)
        self._over_target_box_pose.orientation = Quaternion(0.8, -0.6, 0.008, -0.01)

        self._in_target_box_pose = Pose()
        self._in_target_box_pose.position = Point(0.386, -0.492, 0.233)
        self._in_target_box_pose.orientation = Quaternion(0.8, -0.6, 0.008, -0.01)

        self._over_distractor_box_pose = Pose()
        self._over_distractor_box_pose.position = Point(0.453, 0.474, 0.585)
        self._over_distractor_box_pose.orientation = Quaternion(0.891, 0.45, 0.034, -0.0192)

        self._in_distractor_box_pose = Pose()
        self._in_distractor_box_pose.position = Point(0.453, 0.474, 0.372)
        self._in_distractor_box_pose.orientation = Quaternion(0.891, 0.45, 0.034, -0.0192)

        self._object_poses_in_world_frame = dict()
        self._object_mats_in_world_frame = dict()

        self._lock = Lock()

        self._tf_listener = tf.TransformListener(cache_time=rospy.Duration(1000))
        self._tf_broadcaster = tf.TransformBroadcaster()

    # Object Pose Functions #
    # ----------------------#

    def _get_grasp_pose(self):

        grasp_mat_in_obj_frame = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, self._grasp_overlap],
                                           [0, 0, 0, 1]])

        grasp_mat_in_world_frame = np.matmul(self._object_mats_in_world_frame[self._object_id_to_grasp], grasp_mat_in_obj_frame)
        grasp_pose_in_world_frame = gk.mat_to_quaternion_pose(np, grasp_mat_in_world_frame)
        translation = grasp_pose_in_world_frame[0:3]
        rotation = grasp_pose_in_world_frame[3:]

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

        pre_grasp_mat_in_world_frame = np.matmul(self._object_mats_in_world_frame[self._object_id_to_grasp], pre_grasp_mat_in_obj_frame)
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

        x_offset = 0.2

        z0 = 0

        x1 = 0.15
        y1 = 0.15
        z1 = -0.1

        angle1 = math.pi/8

        # define robot poses
        robot_position_offsets = [np.array([x_offset, 0, z0]),

                                  np.array([x_offset, 0, z1])]#,
        '''
                                  np.array([x_offset+ x1, 0, z1]),
                                  np.array([x_offset + x1, y1, z1]),
                                  np.array([x_offset, y1, z1]),
                                  np.array([x_offset-x1, y1, z1]),
                                  np.array([x_offset-x1, 0, z1]),
                                  np.array([x_offset-x1, -y1, z1]),
                                  np.array([x_offset, -y1, z1]),
                                  np.array([x_offset + x1, -y1, z1])]'''

        robot_rotation_vectors = [np.array([0, 0, 0]),

                                  np.array([0, 0, 0])]#,
        '''
                                  np.array([0, 1, 0])*angle1,
                                  np.array([-0.5**0.5, 0.5**0.5, 0])*angle1,
                                  np.array([-1, 0, 0])*angle1,
                                  np.array([-0.5**0.5, -0.5**0.5, 0])*angle1,
                                  np.array([0, -1, 0])*angle1,
                                  np.array([0.5**0.5, -0.5**0.5, 0])*angle1,
                                  np.array([1, 0, 0])*angle1,
                                  np.array([0.5**0.5, 0.5**0.5, 0])*angle1]'''

        robot_quaternion_offsets = [gk.rotation_vector_to_quaternion(np, aa) for aa in robot_rotation_vectors]

        robot_positions = [self._robot_interface.home_pose[0:3] + offset for offset in robot_position_offsets]

        robot_quaternions = [gk.hamilton_product(np, self._robot_interface.home_pose[3:], qt) for qt in robot_quaternion_offsets]

        self._robot_poses = [np.concatenate((pos, quat),-1) for pos, quat in zip(robot_positions, robot_quaternions)]

    def _initialization_motion(self):
        self._robot_interface.move_to_home(0.025, 0.025)
        for robot_pose in self._robot_poses:

            pose = Pose()
            pose.position = Point(robot_pose[0], robot_pose[1], robot_pose[2])
            pose.orientation = Quaternion(robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6])

            self._robot_interface.set_end_effector_quaternion_pose(pose, 0.025, 0.025)
        self._robot_interface.move_to_home(0.025, 0.025)

    def _move_robot_over_table(self):
        self._robot_interface.move_to_home(0.9, 0.9)

    def _move_robot_to_pre_grasp_pose(self, pre_grasp_pose):
        #self._robot_interface.set_end_effector_quaternion_pointing_pose(pre_grasp_pose)
        self._robot_interface.set_end_effector_quaternion_pose(pre_grasp_pose, 0.9, 0.9)

    def _move_robot_to_grasp_pose(self, grasp_pose):
        self._robot_interface.set_end_effector_position(grasp_pose.position, 0.25, 0.25)

    def _move_robot_to_post_grasp_pose(self, post_grasp_pose):
        self._robot_interface.set_end_effector_position(post_grasp_pose.position, 0.25, 0.25)

    def _suction_grip_object(self):
        time.sleep(2)

    def _move_robot_over_target_box(self):
        self._robot_interface.set_end_effector_quaternion_pose(self._over_target_box_pose, 0.9, 0.9)

    def _move_robot_over_distractor_box(self):
        self._robot_interface.set_end_effector_quaternion_pose(self._over_distractor_box_pose, 0.9, 0.9)

    def _move_robot_in_target_box(self):
        self._robot_interface.set_end_effector_position(self._in_target_box_pose.position, 0.9, 0.9)

    def _move_robot_in_distractor_box(self):
        self._robot_interface.set_end_effector_position(self._in_distractor_box_pose.position, 0.9, 0.9)

    def _release_suction_grip(self):
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

    def _object_pose_callback(self, object_poses):

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
            object_pose_array_in_world_frame = np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
            self._object_poses_in_world_frame[object_pose.class_id] = (object_pose_array_in_world_frame)

            object_mat_in_world_frame = gk.quaternion_pose_to_mat(np, object_pose_array_in_world_frame)
            self._object_mats_in_world_frame[object_pose.class_id] = (object_mat_in_world_frame)

        self._ordered_object_ids_to_grasp = [object_pose.class_id for object_pose in object_poses.poses]

        print('object tree:')
        for class_id in self._ordered_object_ids_to_grasp:
            print(ycb_video_dataset.class_names[class_id])

    def run(self):

        print('performing initialization motion...')
        self._initialization_motion()

        print('waiting for object tree')
        try:
            object_poses = rospy.wait_for_message('/camera/select_picking_order/output/poses', ObjectPoseArray, timeout=15)
        except:
            raise Exception('Object Tree Not Found.')
        self._object_pose_callback(object_poses)

        print('initialization complete')

        while not self._all_objects_removed:

            self._object_id_to_grasp = self._choose_next_object_to_grasp()
            translation = self._object_poses_in_world_frame[self._object_id_to_grasp][0:3]
            rotation = self._object_poses_in_world_frame[self._object_id_to_grasp][3:]
            self._tf_broadcaster.sendTransform(translation, rotation, self._object_ros_time, 'object', 'panda_link0')

            pre_grasp_pose = self._get_pre_grasp_pose()
            grasp_pose = self._get_grasp_pose()
            post_grasp_pose = pre_grasp_pose

            print('picking up ' + str(ycb_video_dataset.class_names[self._object_id_to_grasp]))

            self._move_robot_over_table()
            self._move_robot_to_pre_grasp_pose(pre_grasp_pose)
            self._move_robot_to_grasp_pose(grasp_pose)
            self._suction_grip_object()
            self._move_robot_to_post_grasp_pose(post_grasp_pose)

            self._move_robot_over_table()
            if self._all_distractors_removed:
                self._move_robot_over_target_box()
                self._move_robot_in_target_box()
            else:
                self._move_robot_over_distractor_box()
                self._move_robot_in_distractor_box()
            self._release_suction_grip()

            self._move_robot_over_table()
            self._picked_objects.append(self._object_id_to_grasp)
            self._all_objects_removed = self._check_if_all_objects_removed()
            self._all_distractors_removed = self._check_if_all_distractors_removed()
            print('pick completed')

        print('Demo completed!')


def main():
    robot_demo = RobotDemo()
    input('press enter to continue demo')
    robot_demo.run()


if __name__ == '__main__':
    main()

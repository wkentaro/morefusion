import time
import math
import numpy as np
import robot_demo.general_kinematics as gk
import rospy
import tf
from robot_demo.robot_interface import RobotInterface
from ros_objslampp_msgs.msg import ObjectPoseArray
import objslampp.datasets.ycb_video as ycb_video_dataset


class RobotDemo:

    def __init__(self):

        self._object_models = ycb_video_dataset.YCBVideoModels()

        rospy.init_node('robot_demo')
        self._robot_interface = RobotInterface()
        self._all_objects_collected = False

        self._define_robot_poses()
        self._object_tree_found = False
        self._pose_received = False
        self._object_id_to_grasp = None
        self._picked_objects = list()

        self._link_to_ef_dist = 0.1034

        self._tf_listener = tf.TransformListener()
        self._tf_broadcaster = tf.TransformBroadcaster()


    # Object Pose Functions #
    # ----------------------#

    def _object_grasp_order_callback(self, object_poses):
        self._ordered_object_ids_to_grasp = [object_pose.class_id for object_pose in object_poses.poses]
        print('object tree:')
        for class_id in self._ordered_object_ids_to_grasp:
            print(ycb_video_dataset.class_names[class_id])
        self._object_tree_found = True


    def _object_poses_callback(self, object_poses):

        # decompose pose array
        object_poses_header = object_poses.header
        object_pose = None
        if len(object_poses.poses) == 0:
            return
        for i in range(len(object_poses.poses)):
            object_pose = object_poses.poses[i]
            if object_pose.class_id == self._object_id_to_grasp:
                break
        pose = object_pose.pose
        position = pose.position
        orientation = pose.orientation

        # construct object frame transform
        translation = (position.x, position.y, position.z)
        rotation = (orientation.x, orientation.y, orientation.z, orientation.w)
        ros_time = object_poses_header.stamp
        child_frame = 'object_frame'
        parent_frame = object_poses_header.frame_id

        # publish transform
        self._tf_broadcaster.sendTransform(translation, rotation, ros_time, child_frame, parent_frame)

        # grasping information
        object_class_id = object_pose.class_id
        object_class_name = ycb_video_dataset.class_names[object_class_id]
        grasping_axis = ycb_video_dataset.grasping_axes[object_class_name]

        # publish grasp transform
        translation = [0, 0, 0]
        translation[grasping_axis] =\
            self._object_models.get_cad(class_name=object_class_name).extents[grasping_axis]/2 + self._link_to_ef_dist
        translation = tuple(translation)

        if grasping_axis == 0:
            rotation_vector = np.array([0,-math.pi/2,0])
            rotation = tuple(gk.rotation_vector_to_quaternion(np, rotation_vector))
        elif grasping_axis == 1:
            rotation_vector = np.array([math.pi/2,0,0])
            rotation = tuple(gk.rotation_vector_to_quaternion(np, rotation_vector))
        elif grasping_axis == 2:
            rotation = (0, 0, 0, 1)
        ros_time = object_poses_header.stamp
        child_frame = 'grasp_frame'
        parent_frame = 'object_frame'
        self._tf_broadcaster.sendTransform(translation, rotation, ros_time, child_frame, parent_frame)

        # publish pre grasp transform
        translation = (0, 0, -0.05)
        rotation = (0, 0, 0, 1)
        ros_time = object_poses_header.stamp
        child_frame = 'pre_grasp_frame'
        parent_frame = 'grasp_frame'
        self._tf_broadcaster.sendTransform(translation, rotation, ros_time, child_frame, parent_frame)

        self._pose_received = True

    def _get_pose(self, frame_name):
        common_time = None
        while True:
            try:
                common_time = self._tf_listener.getLatestCommonTime('map', frame_name)
                break
            except:
                continue
        object_transform_tuple = self._tf_listener.lookupTransform('map', frame_name, common_time)
        object_position = object_transform_tuple[0]
        object_orientation = object_transform_tuple[1]
        return np.concatenate((np.array(object_position), np.array(object_orientation)), -1)

    def _get_pre_grasp_pose(self):
        return self._get_pose('pre_grasp_frame')

    def _get_grasp_pose(self):
        return self._get_pose('grasp_frame')

    # Robot Functions #
    # ----------------#

    def _define_robot_poses(self):

        x_offset = 0

        z0 = 0

        x1 = 0.15
        y1 = 0.15
        z1 = -0.1
        angle1 = math.pi/8

        x2 = 0.1
        y2 = 0.1
        z2 = -0.05
        angle2 = math.pi/12

        z3 = -0.2

        # define robot poses
        robot_position_offsets = [np.array([x_offset, 0, z0]),

                                  np.array([x_offset, 0, z1]),
                                  np.array([x_offset+ x1, 0, z1]),
                                  np.array([x_offset + x1, y1, z1]),
                                  np.array([x_offset, y1, z1]),
                                  np.array([x_offset-x1, y1, z1]),
                                  np.array([x_offset-x1, 0, z1]),
                                  np.array([x_offset-x1, -y1, z1]),
                                  np.array([x_offset, -y1, z1]),
                                  np.array([x_offset + x1, -y1, z1])]#,
        '''
                                  np.array([x_offset, 0, z2]),
                                  np.array([x_offset + x2, 0, z2]),
                                  np.array([x_offset + x2, y2, z2]),
                                  np.array([x_offset, y2, z2]),
                                  np.array([x_offset-x2, y2, z2]),
                                  np.array([x_offset-x2, 0, z2]),
                                  np.array([x_offset-x2, -y2, z2]),
                                  np.array([x_offset, -y2, z2]),
                                  np.array([x_offset + x2, -y2, z2]),

                                  np.array([x_offset, 0, z3])]'''

        robot_rotation_vectors = [np.array([0, 0, 0]),

                                  np.array([0, 0, 0]),
                                  np.array([0, 1, 0])*angle1,
                                  np.array([-0.5**0.5, 0.5**0.5, 0])*angle1,
                                  np.array([-1, 0, 0])*angle1,
                                  np.array([-0.5**0.5, -0.5**0.5, 0])*angle1,
                                  np.array([0, -1, 0])*angle1,
                                  np.array([0.5**0.5, -0.5**0.5, 0])*angle1,
                                  np.array([1, 0, 0])*angle1,
                                  np.array([0.5**0.5, 0.5**0.5, 0])*angle1]#,
        '''
                                  np.array([0, 0, 0]),
                                  np.array([0, 1, 0]) * angle2,
                                  np.array([-0.5 ** 0.5, 0.5 ** 0.5, 0]) * angle2,
                                  np.array([-1, 0, 0]) * angle2,
                                  np.array([-0.5 ** 0.5, -0.5 ** 0.5, 0]) * angle2,
                                  np.array([0, -1, 0]) * angle2,
                                  np.array([0.5 ** 0.5, -0.5 ** 0.5, 0]) * angle2,
                                  np.array([1, 0, 0]) * angle2,
                                  np.array([0.5 ** 0.5, 0.5 ** 0.5, 0]) * angle2,

                                  np.array([0, 0, 0])]'''

        robot_quaternion_offsets = [gk.rotation_vector_to_quaternion(np, aa) for aa in robot_rotation_vectors]

        robot_positions = [self._robot_interface.home_pose[0:3] + offset for offset in robot_position_offsets]

        robot_quaternions = [gk.hamilton_product(np, self._robot_interface.home_pose[3:], qt) for qt in robot_quaternion_offsets]

        self._robot_poses = [np.concatenate((pos, quat),-1) for pos, quat in zip(robot_positions, robot_quaternions)]

    def _initialization_motion(self):
        self._robot_interface.move_to_home()
        for robot_pose in self._robot_poses:
            self._robot_interface.set_end_effector_quaternion_pose(robot_pose)

    def _move_robot_over_table(self):
        self._robot_interface.move_to_home()

    def _move_robot_to_pre_grasp_pose(self, pre_grasp_pose):
        pre_grasp_pose = pre_grasp_pose.copy()
        self._robot_interface.set_end_effector_quaternion_pose(pre_grasp_pose)

    def _move_robot_until_force_feedback(self, grasp_pose):
        grasp_pose = grasp_pose.copy()
        self._robot_interface.set_end_effector_quaternion_pose(grasp_pose)

    def _suction_grip_object(self):
        pass

    def _place_object_on_table(self):
        pass

    # Object Checking #
    # ----------------#

    def _check_if_all_objects_collected(self):
        if len(self._picked_objects) < len(self._ordered_object_ids_to_grasp):
            return False
        return True

    def run(self):
        #self._initialization_motion()

        self._obj_grasp_order_sub = rospy.Subscriber('/camera/select_picking_order/output/poses', ObjectPoseArray, self._object_grasp_order_callback)
        self._obj_poses_sub = rospy.Subscriber('/camera/with_occupancy/collision_based_pose_refinement/object_mapping/output/poses', ObjectPoseArray, self._object_poses_callback)

        print('waiting for object tree')
        while not self._object_tree_found:
            continue
        print('computed object tree')

        while not self._all_objects_collected:

            for id in self._ordered_object_ids_to_grasp:
                if id in self._picked_objects:
                    continue
                self._object_id_to_grasp = id
                break
            self._pose_received = False
            while not self._pose_received:
                continue

            pre_grasp_pose = self._get_pre_grasp_pose()
            grasp_pose = self._get_grasp_pose()
            print('picking up ' + str(ycb_video_dataset.class_names[self._object_id_to_grasp]))
            self._move_robot_over_table()
            self._move_robot_to_pre_grasp_pose(pre_grasp_pose)
            self._move_robot_until_force_feedback(grasp_pose)
            self._suction_grip_object()
            self._move_robot_to_pre_grasp_pose(pre_grasp_pose)
            self._place_object_on_table()
            self._move_robot_over_table()
            self._picked_objects.append(self._object_id_to_grasp)
            self._all_objects_collected = self._check_if_all_objects_collected()
            print('pick completed')

        print('Demo completed!')


def main():
    robot_demo = RobotDemo()
    robot_demo.run()


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import io

import networkx
import numpy as np
import PIL.Image
import pybullet
import trimesh
import trimesh.transformations as ttf

import objslampp

import cv_bridge
import message_filters
import tf
import topic_tools
import rospy
from ros_objslampp_msgs.msg import ObjectPoseArray
from sensor_msgs.msg import CameraInfo, Image


def get_leaves(graph, target):
    edges = graph.edges(target)
    if not edges:
        yield target

    for u, v in edges:
        yield from get_leaves(graph, v)


def get_picking_order(graph, target):
    graph = graph.copy()

    order = []
    while True:
        leaves = set(list(get_leaves(graph, target)))
        if target in leaves:
            order.append(target)
            break
        for leaf in leaves:
            order.append(leaf)
            graph.remove_node(leaf)
    return order


def nx_graph_to_image(graph, dpi=300):
    with io.BytesIO() as f:
        agraph = networkx.nx_agraph.to_agraph(graph)
        agraph.graph_attr.update(dpi=300)
        agraph.layout(prog='dot')
        agraph.draw(f, format='png')
        img = np.asarray(PIL.Image.open(f))[:, :, :3]
        return img


def get_mask_center(shape, edge_ratio=0.15):
    H, W = shape[:2]
    mask_center = np.zeros((H, W), dtype=bool)
    x1 = int(round(W * 0.15))
    x2 = W - x1
    y1 = int(round(H * 0.15))
    y2 = H - y1
    mask_center[y1:y2, x1:x2] = True
    return mask_center


class SelectPickingOrder(topic_tools.LazyTransport):

    _models = objslampp.datasets.YCBVideoModels()

    def __init__(self):
        super().__init__()
        self._target = rospy.get_param('~target')
        self._pub_poses = self.advertise(
            '~output/poses', ObjectPoseArray, queue_size=1
        )
        self._pub_graph = self.advertise('~output/graph', Image, queue_size=1)
        self._tf_listener = tf.TransformListener(cache_time=rospy.Duration(30))
        self._post_init()

    def subscribe(self):
        sub_cam = message_filters.Subscriber(
            '~input/camera_info', CameraInfo, queue_size=1
        )
        sub_poses = message_filters.Subscriber(
            '~input/poses', ObjectPoseArray, queue_size=1
        )
        self._subscribers = [sub_cam, sub_poses]
        sync = message_filters.TimeSynchronizer(
            self._subscribers, queue_size=100
        )
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _get_transform(self, source_frame, target_frame, time):
        try:
            self._tf_listener.waitForTransform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=time,
                timeout=rospy.Duration(0.1),
            )
        except Exception as e:
            rospy.logerr(e)
            return

        translation, quaternion = self._tf_listener.lookupTransform(
            target_frame=target_frame,
            source_frame=source_frame,
            time=time,
        )
        translation = np.asarray(translation)
        quaternion = np.asarray(quaternion)[[3, 0, 1, 2]]
        transform = objslampp.functions.transformation_matrix(
            quaternion, translation
        ).array
        return transform

    def _render_object_pose_array(self, cam_msg, poses_msg, instance_id=None):
        depth = np.full((cam_msg.height, cam_msg.width), np.nan, np.float32)
        ins = np.full((cam_msg.height, cam_msg.width), -1, np.int32)
        if not poses_msg.poses:
            return depth, ins

        T_pose2cam = None
        if poses_msg.header.frame_id != cam_msg.header.frame_id:
            T_pose2cam = self._get_transform(
                poses_msg.header.frame_id,
                cam_msg.header.frame_id,
                poses_msg.header.stamp,
            )

        pybullet.connect(pybullet.DIRECT)

        uniq_id_to_ins_id = {}
        for pose in poses_msg.poses:
            if instance_id is not None:
                if pose.instance_id != instance_id:
                    continue

            quaternion, translation = objslampp.ros.from_ros_pose(
                pose.pose
            )
            if T_pose2cam is not None:
                T_cad2pose = objslampp.functions.transformation_matrix(
                    quaternion, translation
                ).array
                T_cad2cam = T_pose2cam @ T_cad2pose
                quaternion = ttf.quaternion_from_matrix(T_cad2cam)
                translation = ttf.translation_from_matrix(T_cad2cam)
            quaternion = quaternion[[1, 2, 3, 0]]

            class_id = pose.class_id
            cad_file = self._models.get_cad_file(class_id=class_id)
            uniq_id = objslampp.extra.pybullet.add_model(
                visual_file=cad_file,
                position=translation,
                orientation=quaternion,
                register=False,
            )
            uniq_id_to_ins_id[uniq_id] = pose.instance_id

        K = np.array(cam_msg.K).reshape(3, 3)
        camera = trimesh.scene.Camera(
            resolution=(cam_msg.width, cam_msg.height),
            focal=(K[0, 0], K[1, 1]),
        )
        _, depth, uniq = objslampp.extra.pybullet.render_camera(
            np.eye(4),
            fovy=camera.fov[1],
            height=camera.resolution[1],
            width=camera.resolution[0],
        )

        pybullet.disconnect()

        for uniq_id, ins_id in uniq_id_to_ins_id.items():
            ins[uniq == uniq_id] = ins_id

        return depth, ins

    def _callback(self, cam_msg, poses_msg):
        bridge = cv_bridge.CvBridge()

        ins_id_to_pose = {}
        class_ids = []
        for pose in poses_msg.poses:
            ins_id_to_pose[pose.instance_id] = pose
            class_ids.append(pose.class_id)
        if self._target not in class_ids:
            return
        del class_ids

        depth_rend, ins_rend = self._render_object_pose_array(
            cam_msg, poses_msg
        )

        target_node_id = None
        graph = networkx.DiGraph()
        class_names = objslampp.datasets.ycb_video.class_names
        mask_center = get_mask_center((cam_msg.height, cam_msg.width))
        for ins_id_i in ins_id_to_pose:
            mask_i = ins_rend == ins_id_i
            if (mask_i & mask_center).sum() < (mask_i & ~mask_center).sum():
                continue

            _, ins = self._render_object_pose_array(
                cam_msg, poses_msg, instance_id=ins_id_i
            )
            mask_whole = ins == ins_id_i
            mask_visible = ins_rend == ins_id_i
            mask_occluded = mask_whole & ~mask_visible

            cls_id_i = ins_id_to_pose[ins_id_i].class_id
            class_name_i = class_names[cls_id_i]
            id_i = (ins_id_i, cls_id_i, class_name_i)
            if cls_id_i == self._target:
                target_node_id = id_i
                graph.add_node(id_i, style='filled', fillcolor='red')
            else:
                graph.add_node(id_i)

            occluded_by, counts = np.unique(
                ins_rend[mask_occluded], return_counts=True
            )

            for ins_id_j, count in zip(occluded_by, counts):
                ratio = count / mask_whole.sum()
                if ratio < 0.01:
                    continue

                cls_id_j = ins_id_to_pose[ins_id_j].class_id
                class_name_j = class_names[cls_id_j]
                id_j = (ins_id_j, cls_id_j, class_name_j)
                graph.add_edge(id_i, id_j)

        poses_msg.poses = []
        if target_node_id is not None:
            order = get_picking_order(graph, target=target_node_id)
            for ins_id, _, _ in order:
                poses_msg.poses.append(ins_id_to_pose[ins_id])
        self._pub_poses.publish(poses_msg)

        img = nx_graph_to_image(graph)
        img_msg = bridge.cv2_to_imgmsg(img, 'rgb8')
        img_msg.header = cam_msg.header
        self._pub_graph.publish(img_msg)


if __name__ == '__main__':
    rospy.init_node('select_picking_order')
    SelectPickingOrder()
    rospy.spin()

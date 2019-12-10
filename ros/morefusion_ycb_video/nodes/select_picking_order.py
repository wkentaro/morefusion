#!/usr/bin/env python

import copy
import io

import imgviz
import networkx
import numpy as np
import PIL.Image
import pybullet
import skimage.segmentation
import trimesh
import trimesh.transformations as ttf

import objslampp

import cv_bridge
import tf
import topic_tools
import rospy
from geometry_msgs.msg import PoseArray
from ros_objslampp_ycb_video.msg import ObjectPoseArray
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
        self._poses_msg = None

        super().__init__()
        self._target = rospy.get_param('~target')
        self._pub_poses = self.advertise(
            '~output/poses', ObjectPoseArray, queue_size=1
        )
        self._pub_poses_viz = self.advertise(
            '~output/poses_viz', PoseArray, queue_size=1
        )
        self._pub_graph = self.advertise(
            '~output/graph', Image, queue_size=1
        )
        self._pub_rend = self.advertise(
            '~output/rgb_rend', Image, queue_size=1
        )
        self._tf_listener = tf.TransformListener(cache_time=rospy.Duration(30))
        self._post_init()

    def subscribe(self):
        self._sub_cam = rospy.Subscriber(
            '~input/camera_info', CameraInfo, self._callback, queue_size=1
        )
        self._sub_poses = rospy.Subscriber(
            '~input/poses', ObjectPoseArray, self._callback_poses, queue_size=1
        )
        self._subscribers = [self._sub_cam, self._sub_poses]

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
        rgb = np.zeros((cam_msg.height, cam_msg.width, 3), np.uint8)
        depth = np.full((cam_msg.height, cam_msg.width), np.nan, np.float32)
        ins = np.full((cam_msg.height, cam_msg.width), -1, np.int32)
        if not poses_msg.poses:
            return rgb, depth, ins

        T_pose2cam = None
        if poses_msg.header.frame_id != 'map':
            raise ValueError('poses_msg.header.frame_id is not "map"')
        T_pose2cam = self._get_transform(
            poses_msg.header.frame_id,
            cam_msg.header.frame_id,
            cam_msg.header.stamp,
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
        rgb, depth, uniq = objslampp.extra.pybullet.render_camera(
            np.eye(4),
            fovy=camera.fov[1],
            height=camera.resolution[1],
            width=camera.resolution[0],
        )

        pybullet.disconnect()

        for uniq_id, ins_id in uniq_id_to_ins_id.items():
            ins[uniq == uniq_id] = ins_id

        return rgb, depth, ins

    def _callback_poses(self, poses_msg):
        self._poses_msg = poses_msg

    def _callback(self, cam_msg):
        if self._poses_msg is None:
            rospy.logwarn_throttle(10, 'self._poses_msg is not set, skipping')
            return
        poses_msg = copy.deepcopy(self._poses_msg)

        bridge = cv_bridge.CvBridge()

        ins_id_to_pose = {}
        class_ids = []
        for pose in poses_msg.poses:
            ins_id_to_pose[pose.instance_id] = pose
            class_ids.append(pose.class_id)
        if self._target not in class_ids:
            rospy.logwarn_throttle(10, 'target object is not yet found')
            return
        del class_ids

        rgb_rend, _, ins_rend = self._render_object_pose_array(
            cam_msg, poses_msg
        )
        rgb_rend_msg = bridge.cv2_to_imgmsg(rgb_rend, 'rgb8')
        rgb_rend_msg.header = cam_msg.header
        self._pub_rend.publish(rgb_rend_msg)
        del rgb_rend

        target_node_id = None
        graph = networkx.DiGraph()
        class_names = objslampp.datasets.ycb_video.class_names
        # mask_center = get_mask_center((cam_msg.height, cam_msg.width))
        K = np.array(cam_msg.K).reshape(3, 3)
        for ins_id_i in np.unique(ins_rend):
            if ins_id_i == -1:
                continue

            # mask_i = ins_rend == ins_id_i
            # if (mask_i & mask_center).sum() < (mask_i & ~mask_center).sum():
            #     continue

            rgb, depth, ins = self._render_object_pose_array(
                cam_msg, poses_msg, instance_id=ins_id_i
            )
            mask_whole = ins == ins_id_i
            mask_visible = ins_rend == ins_id_i
            mask_occluded = mask_whole & ~mask_visible

            pcd = objslampp.geometry.pointcloud_from_depth(
                depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
            )
            quaternion, translation = get_grasp_pose(rgb, pcd, mask_whole)
            ins_id_to_pose[ins_id_i].pose.orientation.w = quaternion[0]
            ins_id_to_pose[ins_id_i].pose.orientation.x = quaternion[1]
            ins_id_to_pose[ins_id_i].pose.orientation.y = quaternion[2]
            ins_id_to_pose[ins_id_i].pose.orientation.z = quaternion[3]
            ins_id_to_pose[ins_id_i].pose.position.x = translation[0]
            ins_id_to_pose[ins_id_i].pose.position.y = translation[1]
            ins_id_to_pose[ins_id_i].pose.position.z = translation[2]

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
                cls_id_j = ins_id_to_pose[ins_id_j].class_id
                rospy.loginfo(f'{cls_id_i} is occluded by {cls_id_j} with occlusion ratio: {ratio}')  # NOQA

                if ratio < 0.1:
                    continue

                class_name_j = class_names[cls_id_j]
                id_j = (ins_id_j, cls_id_j, class_name_j)
                graph.add_edge(id_i, id_j)

        poses_msg = ObjectPoseArray()
        poses_msg.header = cam_msg.header
        poses_viz_msg = PoseArray()
        poses_viz_msg.header = cam_msg.header
        if target_node_id is not None:
            order = get_picking_order(graph, target=target_node_id)
            for ins_id, _, _ in order:
                pose = ins_id_to_pose[ins_id]
                poses_msg.poses.append(pose)
                poses_viz_msg.poses.append(pose.pose)
        self._pub_poses.publish(poses_msg)
        self._pub_poses_viz.publish(poses_viz_msg)

        img = nx_graph_to_image(graph)
        img_msg = bridge.cv2_to_imgmsg(img, 'rgb8')
        img_msg.header = cam_msg.header
        self._pub_graph.publish(img_msg)


def get_grasp_pose(rgb, pcd, mask):
    y1, x1, y2, x2 = imgviz.instances.mask_to_bbox(
        [mask]
    )[0].round().astype(int)

    lbl = np.full(rgb.shape[:2], -1, dtype=np.int32)
    lbl[y1:y2, x1:x2] = skimage.segmentation.slic(
        rgb[y1:y2, x1:x2], n_segments=30, slic_zero=True
    )
    lbl[~mask] = -1
    props = skimage.measure.regionprops(lbl)

    labels = []
    centroids = []
    for prop in props:
        if prop.label == -1:
            continue
        labels.append(prop.label)
        centroids.append(prop.centroid)
    centroid_avg = np.mean(centroids, axis=0)
    index = np.argmin(np.linalg.norm(centroids - centroid_avg, axis=1))
    label = labels[index]

    normals = np.full_like(pcd, np.nan)
    normals[y1:y2, x1:x2] = objslampp.geometry.estimate_pointcloud_normals(
        pcd[y1:y2, x1:x2]
    )

    mask = lbl == label
    translation = np.nanmean(pcd[mask], axis=0)
    normal = np.nanmean(normals[mask], axis=0)
    quaternion = quaternion_from_two_vectors(np.array([0, 0, 1]), normal)
    return quaternion, translation


def quaternion_from_two_vectors(v1, v2):
    v3 = np.cross(v1, v2)
    x, y, z = v3
    w = np.sqrt(
        np.linalg.norm(v1) ** 2 + np.linalg.norm(v2) ** 2
    ) + np.dot(v1, v2)
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


if __name__ == '__main__':
    rospy.init_node('select_picking_order')
    SelectPickingOrder()
    rospy.spin()

#!/usr/bin/env python

import numpy as np
import pybullet
import trimesh
import trimesh.transformations as ttf

import morefusion

import cv_bridge
import rospy
from sensor_msgs.msg import Image, CameraInfo
import tf
from topic_tools import LazyTransport
from visualization_msgs.msg import Marker, MarkerArray


class RenderMeshMarkers(LazyTransport):

    def __init__(self):
        super().__init__()
        self._markers_msg = None
        self._pub = self.advertise('~output', Image, queue_size=1)
        self._tf_listener = tf.TransformListener(cache_time=rospy.Duration(30))
        self._post_init()

        pybullet.connect(pybullet.DIRECT)
        # key: (marker.id, marker.mesh_resource)
        # value: unique_id
        self._marker_to_unique_id = {}

    def __del__(self):
        pybullet.disconnect()

    def subscribe(self):
        sub_markers = rospy.Subscriber(
            '~input/markers', MarkerArray, self._callback_markers, queue_size=1
        )
        sub_cam = rospy.Subscriber(
            '~input/camera_info', CameraInfo, self._callback, queue_size=1
        )
        self._subscribers = [sub_markers, sub_cam]

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback_markers(self, markers_msg):
        self._markers_msg = markers_msg

    def _get_transform(self, source_frame, target_frame, time):
        try:
            self._tf_listener.waitForTransform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=time,
                timeout=rospy.Duration(1),
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
        transform = morefusion.functions.transformation_matrix(
            quaternion, translation
        ).array
        return transform

    def _render_markers_msg(self, cam_msg, markers_msg):
        if markers_msg is None:
            return

        marker_ids = [marker.id for marker in markers_msg.markers]
        for key in list(self._marker_to_unique_id.keys()):
            marker_id, _ = key
            if marker_id not in marker_ids:
                unique_id = self._marker_to_unique_id.pop(key)
                pybullet.removeBody(unique_id)

        transforms = {}  # (marker's frame_id, stamp): T_marker2cam
        for marker in markers_msg.markers:
            if marker.type != Marker.MESH_RESOURCE:
                continue

            quaternion, translation = morefusion.ros.from_ros_pose(marker.pose)
            if marker.header.frame_id != cam_msg.header.frame_id:
                key = (marker.header.frame_id, marker.header.stamp)
                if key in transforms:
                    T_marker2cam = transforms[key]
                else:
                    T_marker2cam = self._get_transform(
                        marker.header.frame_id,
                        cam_msg.header.frame_id,
                        cam_msg.header.stamp,  # assume world is static
                        # marker.header.frame_id
                    )
                    if T_marker2cam is None:
                        return
                    transforms[key] = T_marker2cam
                T_cad2marker = morefusion.functions.transformation_matrix(
                    quaternion, translation
                ).array
                try:
                    T_cad2cam = T_marker2cam @ T_cad2marker
                except ValueError as e:
                    rospy.logerr(e)
                    return
                quaternion = ttf.quaternion_from_matrix(T_cad2cam)
                translation = ttf.translation_from_matrix(T_cad2cam)
            quaternion = quaternion[[1, 2, 3, 0]]

            key = (marker.id, marker.mesh_resource)
            if key in self._marker_to_unique_id:
                unique_id = self._marker_to_unique_id[key]
                pybullet.resetBasePositionAndOrientation(
                    unique_id,
                    translation,
                    quaternion,
                )
            else:
                mesh_file = marker.mesh_resource[len('file://'):]
                unique_id = morefusion.extra.pybullet.add_model(
                    visual_file=mesh_file,
                    position=translation,
                    orientation=quaternion,
                    register=False,
                )
                self._marker_to_unique_id[key] = unique_id

        K = np.array(cam_msg.K).reshape(3, 3)
        camera = trimesh.scene.Camera(
            resolution=(cam_msg.width, cam_msg.height),
            focal=(K[0, 0], K[1, 1])
        )
        rgb_rend, _, _ = morefusion.extra.pybullet.render_camera(
            np.eye(4),
            fovy=camera.fov[1],
            height=camera.resolution[1],
            width=camera.resolution[0],
        )

        return rgb_rend

    def _callback(self, cam_msg):
        rgb_rend = self._render_markers_msg(cam_msg, self._markers_msg)
        if rgb_rend is None:
            rgb_rend = np.full(
                (cam_msg.height, cam_msg.width, 3), 255, dtype=np.uint8
            )

        bridge = cv_bridge.CvBridge()
        rgb_rend_msg = bridge.cv2_to_imgmsg(rgb_rend, encoding='rgb8')
        rgb_rend_msg.header = cam_msg.header
        self._pub.publish(rgb_rend_msg)


if __name__ == '__main__':
    rospy.init_node('render_mesh_markers')
    RenderMeshMarkers()
    rospy.spin()

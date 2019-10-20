#!/usr/bin/env python

import numpy as np
import pybullet
import trimesh
import trimesh.transformations as ttf

import objslampp

import cv_bridge
import message_filters
import rospy
from sensor_msgs.msg import Image, CameraInfo
import tf
from topic_tools import LazyTransport
from visualization_msgs.msg import Marker, MarkerArray


class RenderMeshMarkers(LazyTransport):

    def __init__(self):
        super().__init__()
        self._pub = self.advertise('~output', Image, queue_size=1)
        self._tf_listener = tf.TransformListener(cache_time=rospy.Duration(30))
        self._post_init()

    def subscribe(self):
        sub_cam = message_filters.Subscriber(
            '~input/camera_info', CameraInfo, queue_size=1
        )
        sub_markers = message_filters.Subscriber(
            '~input/markers', MarkerArray, queue_size=1
        )
        self._subscribers = [sub_cam, sub_markers]
        sync = message_filters.ApproximateTimeSynchronizer(
            self._subscribers, queue_size=100, slop=0.1, allow_headerless=True
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

    def _callback(self, cam_msg, markers_msg):
        pybullet.connect(pybullet.DIRECT)

        transforms = {}  # (marker's frame_id, stamp): T_marker2cam
        for marker in markers_msg.markers:
            if marker.type != Marker.MESH_RESOURCE:
                continue

            quaternion, translation = objslampp.ros.from_ros_pose(marker.pose)
            if marker.header.frame_id != cam_msg.header.frame_id:
                key = (marker.header.frame_id, marker.header.stamp)
                if marker.header.frame_id in transforms:
                    T_marker2cam = transforms[key]
                else:
                    T_marker2cam = self._get_transform(
                        marker.header.frame_id,
                        cam_msg.header.frame_id,
                        marker.header.stamp,
                    )
                    transforms[key] = T_marker2cam
                T_cad2marker = objslampp.functions.transformation_matrix(
                    quaternion, translation
                ).array
                T_cad2cam = T_marker2cam @ T_cad2marker
                quaternion = ttf.quaternion_from_matrix(T_cad2cam)
                translation = ttf.translation_from_matrix(T_cad2cam)
            quaternion = quaternion[[1, 2, 3, 0]]

            mesh_file = marker.mesh_resource[len('file://'):]
            objslampp.extra.pybullet.add_model(
                visual_file=mesh_file,
                position=translation,
                orientation=quaternion,
                register=False,
            )

        K = np.array(cam_msg.K).reshape(3, 3)
        camera = trimesh.scene.Camera(
            resolution=(cam_msg.width, cam_msg.height),
            focal=(K[0, 0], K[1, 1])
        )
        rgb_rend, _, _ = objslampp.extra.pybullet.render_camera(
            np.eye(4),
            fovy=camera.fov[1],
            height=camera.resolution[1],
            width=camera.resolution[0],
        )

        pybullet.disconnect()

        bridge = cv_bridge.CvBridge()
        rgb_rend_msg = bridge.cv2_to_imgmsg(rgb_rend, encoding='rgb8')
        rgb_rend_msg.header = cam_msg.header
        self._pub.publish(rgb_rend_msg)


if __name__ == '__main__':
    rospy.init_node('render_mesh_markers')
    RenderMeshMarkers()
    rospy.spin()

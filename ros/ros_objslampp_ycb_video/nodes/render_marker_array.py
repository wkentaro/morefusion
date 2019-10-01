#!/usr/bin/env python

import numpy as np
import pybullet
import trimesh

import objslampp

import cv_bridge
import message_filters
import rospy
from sensor_msgs.msg import Image, CameraInfo
from topic_tools import LazyTransport
from visualization_msgs.msg import Marker, MarkerArray


class RenderMarkerArray(LazyTransport):

    def __init__(self):
        super().__init__()
        self._pub = self.advertise('~output', Image, queue_size=1)
        self._post_init()

    def subscribe(self):
        sub_cam = message_filters.Subscriber(
            '~input/camera_info', CameraInfo
        )
        sub_markers = message_filters.Subscriber(
            '~input/markers', MarkerArray
        )
        self._subscribers = [sub_cam, sub_markers]
        sync = message_filters.ApproximateTimeSynchronizer(
            self._subscribers, queue_size=100, slop=0.1, allow_headerless=True
        )
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback(self, cam_msg, markers_msg):
        pybullet.connect(pybullet.DIRECT)

        for marker in markers_msg.markers:
            if marker.type != Marker.MESH_RESOURCE:
                continue

            mesh_file = marker.mesh_resource[len('file://'):]
            position = [
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z,
            ]
            orientation = [
                marker.pose.orientation.x,
                marker.pose.orientation.y,
                marker.pose.orientation.z,
                marker.pose.orientation.w,
            ]
            objslampp.extra.pybullet.add_model(
                visual_file=mesh_file,
                position=position,
                orientation=orientation,
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
    rospy.init_node('render_marker_array')
    RenderMarkerArray()
    rospy.spin()

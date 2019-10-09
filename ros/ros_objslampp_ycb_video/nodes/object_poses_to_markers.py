#!/usr/bin/env python

import objslampp

import rospy
import topic_tools
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from ros_objslampp_msgs.msg import ObjectPoseArray


class ObjectPosesToMarkers(topic_tools.LazyTransport):

    _models = objslampp.datasets.YCBVideoModels()

    def __init__(self):
        super().__init__()
        self._pub = self.advertise('~output', MarkerArray, queue_size=1)
        self._post_init()

    def subscribe(self):
        self._sub = rospy.Subscriber(
            '~input',
            ObjectPoseArray,
            self._callback,
            queue_size=1,
        )

    def unsubscribe(self):
        self._sub.unregister()

    def _callback(self, poses_msg):
        markers_msg = MarkerArray()

        marker = Marker(
            header=poses_msg.header,
            id=-1,
            action=Marker.DELETEALL
        )
        markers_msg.markers.append(marker)

        for pose in poses_msg.poses:
            marker = Marker()
            marker.header = poses_msg.header
            marker.id = pose.instance_id
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD
            marker.pose = pose.pose
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            cad_file = self._models.get_cad_file(class_id=pose.class_id)
            marker.mesh_resource = f'file://{cad_file}'
            marker.mesh_use_embedded_materials = True
            markers_msg.markers.append(marker)

        self._pub.publish(markers_msg)


if __name__ == '__main__':
    rospy.init_node('object_poses_to_markers')
    ObjectPosesToMarkers()
    rospy.spin()

#!/usr/bin/env python

import rospy
from tf2_msgs.msg import TFMessage


class RepublishTFStaticForRosbag:
    def __init__(self):
        self._tf_msg = TFMessage()
        self._pub = rospy.Publisher(
            "/tf_static_republished", TFMessage, queue_size=1, latch=True
        )
        self._sub = rospy.Subscriber(
            "/tf_static", TFMessage, self._sub_callback
        )
        self._timer = rospy.Timer(rospy.Duration(1), self._timer_callback)

    def _sub_callback(self, tf_msg):
        for transform in tf_msg.transforms:
            self._tf_msg.transforms.append(transform)
        self._pub.publish(self._tf_msg)

    def _timer_callback(self, event):
        if not self._tf_msg.transforms:
            rospy.logwarn_throttle(10, "tf_msg is empty")
            return
        self._pub.publish(self._tf_msg)


if __name__ == "__main__":
    rospy.init_node("republish_tf_static_for_rosbag")
    RepublishTFStaticForRosbag()
    rospy.spin()

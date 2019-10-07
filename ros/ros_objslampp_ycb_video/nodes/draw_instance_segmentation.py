#!/usr/bin/env python

from topic_tools import LazyTransport

import cv_bridge
from jsk_recognition_msgs.msg import ClassificationResult
import message_filters
import rospy
from sensor_msgs.msg import Image

import imgviz
import numpy as np


class DrawInstanceSegmentation(LazyTransport):

    def __init__(self):
        super().__init__()
        self._pub = self.advertise('~output', Image, queue_size=1)

    def subscribe(self):
        sub_rgb = message_filters.Subscriber('~input/rgb', Image)
        sub_ins = message_filters.Subscriber('~input/label_ins', Image)
        sub_lbl = message_filters.Subscriber(
            '~input/class', ClassificationResult
        )
        self._subscribers = [sub_rgb, sub_ins, sub_lbl]
        sync = message_filters.TimeSynchronizer(
            self._subscribers, queue_size=100
        )
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback(self, rgb_msg, ins_msg, cls_msg):
        bridge = cv_bridge.CvBridge()

        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        ins = bridge.imgmsg_to_cv2(ins_msg)

        class_ids = cls_msg.labels
        class_names = cls_msg.label_names
        class_confs = cls_msg.label_proba
        captions = [
            f'{i}: {n}: {c:.1%}'
            for i, n, c in zip(class_ids, class_names, class_confs)
        ]

        instance_ids = np.arange(0, len(class_ids))
        masks = np.array([ins == i for i in instance_ids])

        if masks.size:
            viz = imgviz.instances.instances2rgb(
                image=rgb,
                masks=masks,
                labels=class_ids,
                captions=captions,
                font_size=15,
            )
        else:
            viz = rgb
        viz_msg = bridge.cv2_to_imgmsg(viz, encoding='rgb8')
        viz_msg.header = rgb_msg.header
        self._pub.publish(viz_msg)


if __name__ == '__main__':
    rospy.init_node('draw_instance_segmentation')
    DrawInstanceSegmentation()
    rospy.spin()

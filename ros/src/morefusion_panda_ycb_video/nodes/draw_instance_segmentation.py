#!/usr/bin/env python

from topic_tools import LazyTransport

import cv_bridge
import message_filters
from morefusion_panda_ycb_video.msg import ObjectClassArray
import rospy
from sensor_msgs.msg import Image

import imgviz
import numpy as np

import morefusion


class DrawInstanceSegmentation(LazyTransport):

    _class_names = morefusion.datasets.ycb_video.class_names

    def __init__(self):
        super().__init__()
        self._pub = self.advertise("~output", Image, queue_size=1)

    def subscribe(self):
        sub_rgb = message_filters.Subscriber("~input/rgb", Image)
        sub_ins = message_filters.Subscriber("~input/label_ins", Image)
        sub_lbl = message_filters.Subscriber("~input/class", ObjectClassArray)
        self._subscribers = [sub_rgb, sub_ins, sub_lbl]
        if rospy.get_param("approximate_sync", False):
            sync = message_filters.ApproximateTimeSynchronizer(
                self._subscribers, queue_size=100, slop=0.1
            )
        else:
            sync = message_filters.TimeSynchronizer(
                self._subscribers, queue_size=100
            )
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback(self, rgb_msg, ins_msg, cls_msg):
        bridge = cv_bridge.CvBridge()

        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        ins = bridge.imgmsg_to_cv2(ins_msg)

        instance_ids = []
        class_ids = []
        class_confs = []
        masks = []
        captions = []
        for cls in cls_msg.classes:
            instance_ids.append(cls.instance_id)
            class_ids.append(cls.class_id)
            class_confs.append(cls.confidence)
            masks.append(ins == cls.instance_id)
            class_name = self._class_names[cls.class_id]
            captions.append(
                f"{cls.class_id}: {class_name}: {cls.confidence:.1%}"
            )
        masks = np.array(masks)

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
        viz_msg = bridge.cv2_to_imgmsg(viz, encoding="rgb8")
        viz_msg.header = rgb_msg.header
        self._pub.publish(viz_msg)


if __name__ == "__main__":
    rospy.init_node("draw_instance_segmentation")
    DrawInstanceSegmentation()
    rospy.spin()

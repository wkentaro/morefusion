#!/usr/bin/env python

import cv_bridge
import rospy
from topic_tools import LazyTransport
from sensor_msgs.msg import Image

from chainercv.links.model.fpn import MaskRCNNFPNResNet50
import imgviz
import numpy as np
import objslampp


class MaskRCNNInstanceSegmentationNode(LazyTransport):

    def __init__(self):
        super().__init__()

        self._class_names = objslampp.datasets.ycb_video.class_names

        self._model = MaskRCNNFPNResNet50(
            n_fg_class=len(self._class_names[1:]),
            pretrained_model='/home/wkentaro/objslampp/examples/ycb_video/instance_segm/logs/20190518_071729/model_iter_best',  # NOQA
        )
        self._model.to_gpu()

        self._pub = self.advertise('~output', Image, queue_size=1)

    def subscribe(self):
        self._sub = rospy.Subscriber('~input', Image, callback=self.callback,
                                     queue_size=1, buff_size=2 ** 24)

    def unsubscribe(self):
        self._sub.unregister()

    def callback(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        rgb = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='rgb8')

        with objslampp.utils.timer():
            masks, labels, confs = self._model.predict(
                [rgb.astype(np.float32).transpose(2, 0, 1)]
            )
        masks = masks[0]
        labels = labels[0]
        confs = confs[0]

        keep = masks.sum(axis=(1, 2)) > 0
        masks = masks[keep]
        labels = labels[keep]
        confs = confs[keep]

        class_ids = labels + 1

        captions = [
            f'{self._class_names[cid]}: {conf:.1%}'
            for cid, conf in zip(class_ids, confs)
        ]
        viz = imgviz.instances.instances2rgb(
            image=rgb,
            masks=masks,
            labels=class_ids,
            captions=captions,
            font_size=15,
        )

        outmsg = bridge.cv2_to_imgmsg(viz, encoding='rgb8')
        outmsg.header = imgmsg.header
        self._pub.publish(outmsg)


if __name__ == '__main__':
    rospy.init_node('mask_rcnn_instance_segmentation')
    MaskRCNNInstanceSegmentationNode()
    rospy.spin()

#!/usr/bin/env python

import cv_bridge
import rospy
from topic_tools import LazyTransport
from sensor_msgs.msg import Image

from chainercv.links.model.fpn import MaskRCNNFPNResNet50
import gdown
import imgviz
import numpy as np
import objslampp


class MaskRCNNInstanceSegmentationNode(LazyTransport):

    def __init__(self):
        super().__init__()

        self._class_names = objslampp.datasets.ycb_video.class_names

        pretrained_model = gdown.cached_download(
            url='https://drive.google.com/uc?id=1Ge2S9JudxC5ODdsrjOy5XoW7l7Zcz65E',  # NOQA
            md5='fc06b1292a7e99f9c1deb063accbf7ea',
        )
        self._model = MaskRCNNFPNResNet50(
            n_fg_class=len(self._class_names[1:]),
            pretrained_model=pretrained_model,
        )
        self._model.to_gpu()

        self._pub_ins = self.advertise(
            '~output/label_ins', Image, queue_size=1
        )
        self._pub_cls = self.advertise(
            '~output/label_cls', Image, queue_size=1
        )
        self._pub_viz = self.advertise('~output/viz', Image, queue_size=1)
        self._post_init()

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

        label_ins = np.full(rgb.shape[:2], -1, dtype=np.int32)
        label_cls = np.zeros(rgb.shape[:2], dtype=np.int32)
        sort = np.argsort(confs)
        for ins_id, (cls_id, mask) in enumerate(
            zip(class_ids[sort], masks[sort])
        ):
            label_ins[mask] = ins_id
            label_cls[mask] = cls_id

        outmsg = bridge.cv2_to_imgmsg(label_ins)
        outmsg.header = imgmsg.header
        self._pub_ins.publish(outmsg)
        outmsg = bridge.cv2_to_imgmsg(label_cls)
        outmsg.header = imgmsg.header
        self._pub_cls.publish(outmsg)

        if self._pub_viz.get_num_connections() > 0:
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
            self._pub_viz.publish(outmsg)


if __name__ == '__main__':
    rospy.init_node('mask_rcnn_instance_segmentation')
    MaskRCNNInstanceSegmentationNode()
    rospy.spin()

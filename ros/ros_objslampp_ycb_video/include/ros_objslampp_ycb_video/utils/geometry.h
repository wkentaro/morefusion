// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_

#include <vector>
#include <map>
#include <utility>

#include <opencv2/opencv.hpp>

#include "ros_objslampp_ycb_video/utils/opencv.h"


namespace ros_objslampp_ycb_video {
namespace utils {

void track_instance_id(
    const cv::Mat& reference,
    cv::Mat* target,
    std::map<int, unsigned>* instance_id_to_class_id) {
  std::vector<int> instance_ids1 = ros_objslampp_ycb_video::utils::unique<int>(reference);
  std::vector<int> instance_ids2 = ros_objslampp_ycb_video::utils::unique<int>(*target);

  std::map<int, std::pair<int, float> > ins_id1to2;
  for (size_t i = 0; i < instance_ids1.size(); i++) {
    // ins_id1: instance_id in the map
    int ins_id1 = instance_ids1[i];
    if (ins_id1 == -1) {
      continue;
    }

    cv::Mat mask1 = reference == ins_id1;
    for (size_t j = 0; j < instance_ids2.size(); j++) {
      // ins_id2: instance_id in the mask-rcnn output
      int ins_id2 = instance_ids2[j];
      if (ins_id2 == -1) {
        continue;
      }

      cv::Mat mask2 = (*target) == ins_id2;
      cv::Mat mask_intersection, mask_union;
      cv::bitwise_and(mask1, mask2, mask_intersection);
      cv::bitwise_or(mask1, mask2, mask_union);
      float iou = (cv::sum(mask_intersection) / cv::sum(mask_union))[0];
      std::map<int, std::pair<int, float> >::iterator it2 = ins_id1to2.find(ins_id1);
      if ((it2 == ins_id1to2.end())) {
        ins_id1to2.insert(std::make_pair(ins_id1, std::make_pair(ins_id2, iou)));
      } else if (iou > it2->second.second) {
        it2->second = std::make_pair(ins_id2, iou);
      }
    }
  }

  for (std::map<int, unsigned>::iterator it = instance_id_to_class_id->begin();
       it != instance_id_to_class_id->end(); it++) {
    int ins_id1 = it->first;
    unsigned class_id = it->second;
    int ins_id2 = ins_id1to2.find(ins_id1)->second.first;
    if (ins_id1 != ins_id2) {
      instance_id_to_class_id->erase(it);
      instance_id_to_class_id->insert(std::make_pair(ins_id1, class_id));
    }
  }

  for (size_t j = 0; j < target->rows; j++) {
    for (size_t i = 0; i < target->cols; i++) {
      for (std::map<int, std::pair<int, float> >::iterator it = ins_id1to2.begin();
           it != ins_id1to2.end(); it++) {
        int ins_id1 = it->first;
        int ins_id2 = it->second.first;
        if (ins_id1 == ins_id2) {
          continue;
        }
        if (target->at<int>(j, i) == ins_id2) {
          target->at<int>(j, i) = ins_id1;
          break;
        }
      }
    }
  }
}

}  // namespace utils

}  // namespace ros_objslampp_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_

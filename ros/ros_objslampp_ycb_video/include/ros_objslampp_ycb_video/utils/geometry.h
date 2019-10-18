// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_

#include <algorithm>
#include <vector>
#include <map>
#include <tuple>
#include <utility>

#include <opencv2/opencv.hpp>

#include "ros_objslampp_ycb_video/utils/opencv.h"


namespace ros_objslampp_ycb_video {
namespace utils {

void track_instance_id(
    const cv::Mat& reference,
    cv::Mat* target,
    std::map<int, unsigned>* instance_id_to_class_id,
    unsigned* instance_counter) {
  std::vector<int> instance_ids1 = ros_objslampp_ycb_video::utils::unique<int>(reference);
  std::vector<int> instance_ids2 = ros_objslampp_ycb_video::utils::unique<int>(*target);

  cv::Mat mask_center = cv::Mat::zeros(reference.rows, reference.cols, CV_8UC1);
  cv::rectangle(
    mask_center,
    cv::Point(reference.cols * 0.2, reference.rows * 0.2),
    cv::Point(reference.cols * 0.8, reference.rows * 0.8),
    /*color=*/255,
    /*thickness=*/CV_FILLED);

  // Compute IOU
  std::map<int, std::pair<int, float> > ins_id2to1;
  std::set<int> ins_ids2_on_edge;
  for (size_t i = 0; i < instance_ids2.size(); i++) {
    // ins_id2: instance_id in the mask-rcnn output
    int ins_id2 = instance_ids2[i];
    if (ins_id2 < 0) {
      continue;
    }

    cv::Mat mask2 = (*target) == ins_id2;
    ins_id2to1.insert(std::make_pair(ins_id2, std::make_pair(-1, 0)));
    cv::Mat mask_intersect;
    cv::bitwise_and(mask_center, mask2, mask_intersect);
    if (cv::countNonZero(mask_intersect) == 0) {
      ins_ids2_on_edge.insert(ins_id2);
    }
    for (size_t j = 0; j < instance_ids1.size(); j++) {
      // ins_id1: instance_id in the map
      int ins_id1 = instance_ids1[j];
      if (ins_id1 < 0) {
        continue;
      }

      cv::Mat mask1 = reference == ins_id1;
      cv::Mat mask_intersection, mask_union;
      cv::bitwise_and(mask1, mask2, mask_intersection);
      cv::bitwise_or(mask1, mask2, mask_union);
      float iou =
        static_cast<float>(cv::countNonZero(mask_intersection)) /
        static_cast<float>(cv::countNonZero(mask_union));
      std::map<int, std::pair<int, float> >::iterator it2 = ins_id2to1.find(ins_id2);
      if (iou > it2->second.second) {
        it2->second = std::make_pair(ins_id1, iou);
      }
    }
  }

  // Initialize new object
  for (std::map<int, std::pair<int, float> >::iterator it = ins_id2to1.begin();
       it != ins_id2to1.end(); it++) {
    int ins_id2 = it->first;
    if (ins_ids2_on_edge.find(ins_id2) != ins_ids2_on_edge.end()) {
      // it's on the edge, so don't initialize
      continue;
    }
    int ins_id1 = it->second.first;
    float iou = it->second.second;
    if (iou < 0.3) {
      it->second.first = *instance_counter;
      // new instance
      (*instance_counter)++;
    }
  }

  std::map<int, unsigned> instance_id_to_class_id_updated;
  for (std::map<int, unsigned>::iterator it = instance_id_to_class_id->begin();
       it != instance_id_to_class_id->end(); it++) {
    int ins_id2 = it->first;
    if (ins_ids2_on_edge.find(ins_id2) != ins_ids2_on_edge.end()) {
      // it's on the edge, so skip inserting instance_id_to_class_id_updated
      continue;
    }
    int ins_id1 = ins_id2to1.find(ins_id2)->second.first;
    unsigned class_id = it->second;
    instance_id_to_class_id_updated.insert(std::make_pair(ins_id1, class_id));
  }
  instance_id_to_class_id->clear();
  for (std::map<int, unsigned>::iterator it = instance_id_to_class_id_updated.begin();
       it != instance_id_to_class_id_updated.end(); it++) {
    instance_id_to_class_id->insert(std::make_pair(it->first, it->second));
  }

  for (size_t j = 0; j < target->rows; j++) {
    for (size_t i = 0; i < target->cols; i++) {
      int ins_id2 = target->at<int>(j, i);
      if (ins_id2 < 0) {
        continue;
      }
      if (ins_ids2_on_edge.find(ins_id2) != ins_ids2_on_edge.end()) {
        // it's on the edge, so copy rendering
        // TODO(wkentaro): copy rendered but don't update occupied space with this,
        // since it's renderd with previous frame.
        target->at<int>(j, i) = reference.at<int>(j, i);
        continue;
      }
      std::map<int, std::pair<int, float> >::iterator it2 = ins_id2to1.find(ins_id2);
      assert(it2 != ins_id2to1.end());
      int ins_id1 = it2->second.first;
      if (ins_id1 != ins_id2) {
        target->at<int>(j, i) = ins_id1;
      }
    }
  }
}

std::map<int, std::tuple<int, int, int, int> > label_ins_to_bboxes(const cv::Mat& label_ins) {
  // aabb: y1, x1, y2, x2
  std::map<int, std::tuple<int, int, int, int> > instance_id_to_aabb;

  int height = label_ins.rows;
  int width = label_ins.cols;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      int instance_id = label_ins.at<int>(j, i);
      if (instance_id < 0) {
        continue;
      }

      int y1, x1, y2, x2;
      if (instance_id_to_aabb.find(instance_id) == instance_id_to_aabb.end()) {
        y1 = std::max(j - 1, 0);
        x1 = std::max(i - 1, 0);
        y2 = std::min(j + 1, height - 1);
        x2 = std::min(i + 1, width - 1);
        instance_id_to_aabb.insert(std::make_pair(instance_id, std::make_tuple(y1, x1, y2, x2)));
      } else {
        std::tuple<int, int, int, int> aabb = instance_id_to_aabb.find(instance_id)->second;
        y1 = std::max(std::min(j - 1, std::get<0>(aabb)), 0);
        x1 = std::max(std::min(i - 1, std::get<1>(aabb)), 0);
        y2 = std::min(std::max(j + 1, std::get<2>(aabb)), height - 1);
        x2 = std::min(std::max(i + 1, std::get<3>(aabb)), width - 1);
        instance_id_to_aabb.find(instance_id)->second = std::make_tuple(y1, x1, y2, x2);
      }
    }
  }

  return instance_id_to_aabb;
}

cv::Mat rendering_mask(const cv::Mat& label_ins) {
  std::map<int, std::tuple<int, int, int, int> > instance_id_to_aabb =
    label_ins_to_bboxes(label_ins);

  cv::Mat mask = cv::Mat::zeros(label_ins.rows, label_ins.cols, CV_8UC1);
  for (std::map<int, std::tuple<int, int, int, int> >::iterator it =
       instance_id_to_aabb.begin(); it != instance_id_to_aabb.end(); it++) {
    std::tuple<int, int, int, int> aabb = it->second;
    int y1 = std::get<0>(aabb);
    int x1 = std::get<1>(aabb);
    int y2 = std::get<2>(aabb);
    int x2 = std::get<3>(aabb);

    // render 10% larger ROI
    float cy = (y1 + y2) / 2.0;
    float cx = (x1 + x2) / 2.0;
    float roi_h = y2 - y1;
    float roi_w = x2 - x1;
    roi_h *= 1.5;
    roi_w *= 1.5;

    y1 = cy - static_cast<int>(std::round(roi_h / 2.0));
    y2 = y1 + static_cast<int>(roi_h);
    x1 = cx - static_cast<int>(std::round(roi_w / 2.0));
    x2 = x1 + static_cast<int>(roi_w);

    cv::rectangle(
      mask,
      cv::Point(x1, y1),
      cv::Point(x2, y2),
      /*color=*/255,
      /*thickness=*/CV_FILLED);
  }

  return mask;
}


}  // namespace utils

}  // namespace ros_objslampp_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_

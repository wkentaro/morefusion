// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_

#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <utility>

#include <opencv2/opencv.hpp>

#include "ros_objslampp_ycb_video/utils/opencv.h"
// #include "ros_objslampp_ycb_video/utils/log.h"


namespace ros_objslampp_ycb_video {
namespace utils {

std::tuple<int, int, int, int> mask_to_bbox(const cv::Mat& mask) {
  int height = mask.rows;
  int width = mask.cols;
  int y1 = height - 1;
  int x1 = width - 1;
  int y2 = 0;
  int x2 = 0;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      if (mask.at<uint8_t>(j, i) != 0) {
        y1 = std::max(std::min(j - 1, y1), 0);
        x1 = std::max(std::min(i - 1, x1), 0);
        y2 = std::min(std::max(j + 1, y2), height - 1);
        x2 = std::min(std::max(i + 1, x2), width - 1);
      }
    }
  }
  return std::make_tuple(y1, x1, y2, x2);
}

bool is_detected_mask_too_small(const cv::Mat& mask2) {
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(
    mask2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  cv::Mat mask2_denoised;
  mask2.copyTo(mask2_denoised);
  for (size_t i = 0; i < contours.size(); i++) {
    if (cv::contourArea(contours[i]) < (20 * 20)) {
      cv::drawContours(mask2_denoised, contours, i, /*color=*/0, /*thickness=*/CV_FILLED);
    }
  }
  std::tuple<int, int, int, int> bbox2 = ros_objslampp_ycb_video::utils::mask_to_bbox(
    mask2_denoised);
  int y1 = std::get<0>(bbox2);
  int x1 = std::get<1>(bbox2);
  int y2 = std::get<2>(bbox2);
  int x2 = std::get<3>(bbox2);
  int bbox_height = y2 - y1;
  int bbox_width = x2 - x1;
  int mask_size = cv::countNonZero(mask2_denoised);
  int bbox_size = bbox_height * bbox_width;
  float mask_ratio_in_bbox = static_cast<float>(mask_size) / static_cast<float>(bbox_size);
  if (mask_size < (80 * 80) ||
      bbox_size < (120 * 120) ||
      bbox_height < 100 ||
      bbox_width < 100) {
    return true;
  }
  return false;
}

void track_instance_id(
    cv::Mat& reference,
    cv::Mat* target,
    std::map<int, unsigned>* instance_id_to_class_id,
    unsigned* instance_counter) {
  std::set<int> instance_ids1 = ros_objslampp_ycb_video::utils::unique<int>(reference);
  std::set<int> instance_ids2 = ros_objslampp_ycb_video::utils::unique<int>(*target);

  cv::Mat mask_nonedge = cv::Mat::zeros(reference.rows, reference.cols, CV_8UC1);
  cv::rectangle(
    mask_nonedge,
    cv::Point(reference.cols * 0.05, reference.rows * 0.05),
    cv::Point(reference.cols * 0.95, reference.rows * 0.95),
    /*color=*/255,
    /*thickness=*/CV_FILLED);
  cv::Mat mask_edge;
  cv::bitwise_not(mask_nonedge, mask_edge);

  // Compute IOU
  std::map<int, std::tuple<int, float, float> > ins_id2to1;  // ins_id1, iou, coverage
  std::set<int> ins_ids1_suspicious;
  std::set<int> ins_ids2_suspicious;
  for (int ins_id2 : instance_ids2) {
    // ins_id2: instance_id in the mask-rcnn output
    if (ins_id2 < 0) {
      continue;
    }

    cv::Mat mask2 = (*target) == ins_id2;
    ins_id2to1.insert(std::make_pair(ins_id2, std::make_tuple(-1, 0, 0)));

    if (ros_objslampp_ycb_video::utils::is_detected_mask_too_small(mask2)) {
      ins_ids2_suspicious.insert(ins_id2);
    }

    {
      // Check if mask2 is not on the edge of the image
      cv::Mat mask_intersect_edge, mask_intersect_nonedge;
      cv::bitwise_and(mask_edge, mask2, mask_intersect_edge);
      cv::bitwise_and(mask_nonedge, mask2, mask_intersect_nonedge);
      if (cv::countNonZero(mask_intersect_edge) > cv::countNonZero(mask_intersect_nonedge)) {
        ins_ids2_suspicious.insert(ins_id2);
      }
    }

    for (int ins_id1 : instance_ids1) {
      // ins_id1: instance_id in the map
      if (ins_id1 < 0) {
        continue;
      }

      cv::Mat mask1 = reference == ins_id1;

      {
        // Check if mask1 is not on the edge of the image
        cv::Mat mask_intersect_edge, mask_intersect_nonedge;
        cv::bitwise_and(mask_edge, mask1, mask_intersect_edge);
        cv::bitwise_and(mask_nonedge, mask1, mask_intersect_nonedge);
        if (cv::countNonZero(mask_intersect_edge) > cv::countNonZero(mask_intersect_nonedge)) {
          ins_ids1_suspicious.insert(ins_id1);
        }
      }

      // IOU between mask1 (from map) and mask2 (from detection)
      cv::Mat mask_intersection, mask_union;
      cv::bitwise_and(mask1, mask2, mask_intersection);
      cv::bitwise_or(mask1, mask2, mask_union);
      float iou =
        static_cast<float>(cv::countNonZero(mask_intersection)) /
        static_cast<float>(cv::countNonZero(mask_union));
      float coverage =
        static_cast<float>(cv::countNonZero(mask_intersection)) /
        static_cast<float>(cv::countNonZero(mask1));
      auto it2 = ins_id2to1.find(ins_id2);
      if (iou > std::get<1>(it2->second)) {
        it2->second = std::make_tuple(ins_id1, iou, coverage);
      }
    }
  }

  // Initialize new object
  for (auto& kv : ins_id2to1) {
    int ins_id2 = kv.first;
    if (ins_ids2_suspicious.find(ins_id2) != ins_ids2_suspicious.end()) {
      // it's on the edge, so don't initialize
      continue;
    }
    int ins_id1 = std::get<0>(kv.second);
    float iou = std::get<1>(kv.second);
    float coverage = std::get<2>(kv.second);
    if ((iou >= 0.4) || (coverage >= 0.9)) {
      continue;
    }
    // create new instance
    std::get<0>(kv.second) = *instance_counter;
    (*instance_counter)++;
  }

  std::map<int, unsigned> instance_id_to_class_id_updated;
  for (std::map<int, unsigned>::iterator it = instance_id_to_class_id->begin();
       it != instance_id_to_class_id->end(); it++) {
    int ins_id2 = it->first;
    if (ins_ids2_suspicious.find(ins_id2) != ins_ids2_suspicious.end()) {
      // it's on the edge, so skip inserting instance_id_to_class_id_updated
      continue;
    }
    int ins_id1 = std::get<0>(ins_id2to1.find(ins_id2)->second);
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
        if (mask_edge.at<uint8_t>(j, i) != 0) {
          target->at<int>(j, i) = -2;
        }
        continue;
      }
      if (ins_ids2_suspicious.find(ins_id2) != ins_ids2_suspicious.end()) {
        target->at<int>(j, i) = -2;
        continue;
      }
      auto it2 = ins_id2to1.find(ins_id2);
      assert(it2 != ins_id2to1.end());
      int ins_id1 = std::get<0>(it2->second);
      if (ins_id1 != ins_id2) {
        target->at<int>(j, i) = ins_id1;
      }
    }
  }

  // Manipulate target
  std::set<int> instance_ids_active = ros_objslampp_ycb_video::utils::unique<int>(*target);
  for (int ins_id : instance_ids_active) {
    if (ins_id < 0) {
      continue;
    }
    cv::Mat mask = (*target) == ins_id;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(
      mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    for (size_t j = 0; j < contours.size(); j++) {
      if (cv::contourArea(contours[j]) < (20 * 20)) {
        cv::drawContours(*target, contours, j, /*color=*/-2, /*thickness=*/CV_FILLED);
      } else {
        cv::drawContours(*target, contours, j, /*color=*/-2, /*thickness=*/10);
      }
    }
  }

  // Manipulate reference
  for (int ins_id : ros_objslampp_ycb_video::utils::unique<int>(reference)) {
     if (ins_id < 0) {
       continue;
     }

    cv::Mat mask = reference == ins_id;

    if (ins_ids1_suspicious.find(ins_id) != ins_ids1_suspicious.end()) {
      reference.setTo(-2, mask);
      continue;
    }

    // Remove boundary
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(
      mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    for (size_t j = 0; j < contours.size(); j++) {
      if (cv::contourArea(contours[j]) < (20 * 20)) {
        cv::drawContours(reference, contours, j, /*color=*/-2, /*thickness=*/CV_FILLED);
      } else {
        cv::drawContours(reference, contours, j, /*color=*/-2, /*thickness=*/10);
      }
    }
  }

  // Merge target and reference for pose estimation
  instance_ids1 = ros_objslampp_ycb_video::utils::unique<int>(*target);
  cv::Mat merged(reference.size(), CV_32SC1, -2);
  for (int ins_id2 : ros_objslampp_ycb_video::utils::unique<int>(reference)) {
    if (ins_id2 < 0) {
      continue;
    }

    cv::Mat mask;
    if (instance_ids1.find(ins_id2) != instance_ids1.end()) {
      mask = *target == ins_id2;
    } else {
      mask = reference == ins_id2;
    }
    merged.setTo(ins_id2, mask);
  }
  merged.copyTo(reference);
}

}  // namespace utils

}  // namespace ros_objslampp_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_

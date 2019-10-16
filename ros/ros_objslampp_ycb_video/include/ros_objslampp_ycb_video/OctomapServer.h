// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_OCTOMAPSERVER_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_OCTOMAPSERVER_H_

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <octomap_ros/conversions.h>
#include <octomap/octomap.h>
#include <octomap/OcTreeKey.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/BoundingBoxQuery.h>
#include <octomap_msgs/conversions.h>
#include <ros/ros.h>
#include <ros_objslampp_msgs/VoxelGridArray.h>
#include <ros_objslampp_msgs/ObjectClassArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/ColorRGBA.h>
#include <std_srvs/Empty.h>
#include <tf/message_filter.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <map>
#include <string>

#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>

#include "ros_objslampp_ycb_video/OctomapServerConfig.h"

namespace ros_objslampp_ycb_video {

class OctomapServer {
 public:
  typedef pcl::PointXYZ PCLPoint;
  typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;
  typedef octomap::OcTree OcTreeT;

  typedef octomap_msgs::GetOctomap OctomapSrv;
  typedef octomap_msgs::BoundingBoxQuery BBXSrv;

  typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::PointCloud2, sensor_msgs::Image, ros_objslampp_msgs::ObjectClassArray>
    ExactSyncPolicy;

  explicit OctomapServer(ros::NodeHandle private_nh_ = ros::NodeHandle("~"));
  virtual ~OctomapServer();
  bool resetSrv(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);  // NOLINT

  virtual void insertCloudCallback(
    const sensor_msgs::PointCloud2ConstPtr& cloud,
    const sensor_msgs::ImageConstPtr& ins_msg,
    const ros_objslampp_msgs::ObjectClassArrayConstPtr& class_msg);

 protected:
  // inline static void updateMinKey(const octomap::OcTreeKey& in, octomap::OcTreeKey* min) {
  //   for (unsigned i = 0; i < 3; ++i) {
  //     (*min)[i] = std::min(in[i], (*min)[i]);
  //   }
  // }

  // inline static void updateMaxKey(const octomap::OcTreeKey& in, octomap::OcTreeKey* max) {
  //   for (unsigned i = 0; i < 3; ++i)
  //    (*max)[i] = std::max(in[i], (*max)[i]);
  // }

  void publishBinaryOctoMap(const ros::Time& rostime = ros::Time::now()) const;
  void publishFullOctoMap(const ros::Time& rostime = ros::Time::now()) const;
  virtual void publishAll(const ros::Time& rostime = ros::Time::now());

  /**
  * @brief update occupancy map with a scan labeled as ground and nonground.
  * The scans should be in the global map frame.
  *
  * @param sensorOrigin origin of the measurements for raycasting
  * @param ground scan endpoints on the ground plane (only clear space)
  * @param nonground all other endpoints (clear up to occupied endpoint)
  */
  virtual void insertScan(
    const tf::Point& sensorOrigin,
    const PCLPointCloud& pc,
    const cv::Mat& label_ins,
    const std::map<int, unsigned>& instance_id_to_class_id);

  /**
  * @brief Find speckle nodes (single occupied voxels with no neighbors). Only works on lowest resolution!
  * @param key
  * @return
  */
  bool isSpeckleNode(const octomap::OcTreeKey& key) const;

  void configCallback(
    const ros_objslampp_ycb_video::OctomapServerConfig& config,
    const uint32_t level);

  void renderOctrees(const Eigen::Matrix4f& sensorToWorld, cv::Mat label_ins, cv::Mat depth);

  ros::NodeHandle m_nh;
  ros::Publisher m_markerPub;
  ros::Publisher m_binaryMapPub;
  ros::Publisher m_fullMapPub;
  ros::Publisher m_pointCloudPub;
  ros::Publisher m_collisionObjectPub;
  ros::Publisher m_mapPub;
  ros::Publisher m_cmapPub;
  ros::Publisher m_fmapPub;
  ros::Publisher m_fmarkerPub;
  ros::Publisher m_gridsPub;
  ros::Publisher m_gridsNoEntryPub;
  ros::Publisher m_labelTrackedPub;
  ros::Publisher m_labelRenderedPub;
  ros::Publisher m_classPub;
  dynamic_reconfigure::Server<ros_objslampp_ycb_video::OctomapServerConfig> m_reconfigSrv;
  message_filters::Subscriber<sensor_msgs::PointCloud2>* m_pointCloudSub;
  message_filters::Subscriber<sensor_msgs::Image>* m_labelInsSub;
  message_filters::Subscriber<ros_objslampp_msgs::ObjectClassArray>* m_classSub;
  tf::MessageFilter<sensor_msgs::PointCloud2>* m_tfPointCloudSub;
  message_filters::Synchronizer<ExactSyncPolicy>* m_sync;
  ros::ServiceServer m_resetService;
  tf::TransformListener m_tfListener;

  std::map<int, OcTreeT*> m_octrees;
  std::map<int, unsigned> m_classIds;
  std::map<int, octomap::point3d> m_centers;
  octomap::KeyRay m_keyRay;  // temp storage for ray casting
  // octomap::OcTreeKey m_updateBBXMin;
  // octomap::OcTreeKey m_updateBBXMax;

  double m_maxRange;
  std::string m_worldFrameId;  // the map frame
  std::string m_sensorFrameId;  // base of the robot for ground plane filtering

  bool m_latchedTopics;

  double m_res;
  double m_probHit;
  double m_probMiss;
  double m_thresMin;
  double m_thresMax;

  unsigned m_treeDepth;
  unsigned m_maxTreeDepth;

  double m_occupancyMinZ;
  double m_occupancyMaxZ;
  double m_minSizeX;
  double m_minSizeY;
  bool m_filterSpeckles;

  bool m_compressMap;

  bool m_groundAsNoEntry;
  bool m_freeAsNoEntry;
};

}  // namespace ros_objslampp_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_OCTOMAPSERVER_H_

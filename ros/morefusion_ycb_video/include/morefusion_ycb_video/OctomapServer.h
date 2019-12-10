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
#include <morefusion_ycb_video/VoxelGridArray.h>
#include <morefusion_ycb_video/ObjectClassArray.h>
#include <morefusion_ycb_video/RenderVoxelGridArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/ColorRGBA.h>
#include <std_srvs/Empty.h>
#include <tf/message_filter.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <map>
#include <set>
#include <string>

#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>

#include "morefusion_ycb_video/OctomapServerConfig.h"
#include "morefusion_ycb_video/utils.h"

namespace morefusion_ycb_video {

class OctomapServer {
 public:
  typedef pcl::PointXYZ PCLPoint;
  typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;
  typedef octomap::OcTree OcTreeT;

  typedef octomap_msgs::GetOctomap OctomapSrv;
  typedef octomap_msgs::BoundingBoxQuery BBXSrv;

  typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::CameraInfo,
    sensor_msgs::Image,
    sensor_msgs::PointCloud2,
    sensor_msgs::Image,
    morefusion_ycb_video::ObjectClassArray> ExactSyncPolicy;

  explicit OctomapServer();
  virtual ~OctomapServer() {}

  virtual void insertCloudCallback(
    const sensor_msgs::CameraInfoConstPtr& camera_info,
    const sensor_msgs::ImageConstPtr& depth_msg,
    const sensor_msgs::PointCloud2ConstPtr& cloud,
    const sensor_msgs::ImageConstPtr& ins_msg,
    const morefusion_ycb_video::ObjectClassArrayConstPtr& class_msg);

 protected:
  void publishBinaryOctoMap(const ros::Time& rostime = ros::Time::now()) const;
  void publishFullOctoMap(const ros::Time& rostime = ros::Time::now()) const;
  virtual void publishAll(const ros::Time& rostime = ros::Time::now());

  void getGridsInWorldFrame(const ros::Time& rostime, morefusion_ycb_video::VoxelGridArray& grids);
  void publishGrids(
      const ros::Time& rostime,
      const Eigen::Matrix4f& sensorToWorld,
      const std::set<int>& instance_ids_active);

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
    const morefusion_ycb_video::OctomapServerConfig& config,
    const uint32_t level);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Publisher pub_binary_map_;
  ros::Publisher pub_full_map_;
  ros::Publisher pub_grids_;
  ros::Publisher pub_grids_noentry_;
  ros::Publisher pub_markers_bg_;
  ros::Publisher pub_markers_fg_;
  ros::Publisher pub_markers_free_;
  ros::Publisher pub_label_rendered_;
  ros::Publisher pub_label_tracked_;
  ros::Publisher pub_class_;

  message_filters::Subscriber<sensor_msgs::CameraInfo>* sub_camera_;
  message_filters::Subscriber<sensor_msgs::Image>* sub_depth_;
  message_filters::Subscriber<sensor_msgs::PointCloud2>* sub_pcd_;
  message_filters::Subscriber<sensor_msgs::Image>* sub_label_ins_;
  message_filters::Subscriber<morefusion_ycb_video::ObjectClassArray>* sub_class_;
  message_filters::Synchronizer<ExactSyncPolicy>* sync_;

  dynamic_reconfigure::Server<morefusion_ycb_video::OctomapServerConfig> server_reconfig_;

  ros::ServiceClient client_render_;

  tf::TransformListener* tf_listener_;

  std::map<int, OcTreeT*> octrees_;
  std::map<int, unsigned> class_ids_;
  std::map<int, octomap::point3d> centers_;
  unsigned instance_counter_;

  // mapping parameters
  double resolution_;
  double max_range_;
  double probability_hit_;
  double probability_miss_;
  double probability_min_;
  double probability_max_;
  unsigned tree_depth_;
  unsigned tree_depth_max_;
  bool do_compress_map_;

  // for publishing
  std::string frame_id_world_;
  std::string frame_id_sensor_;
  bool m_groundAsNoEntry;
  bool m_freeAsNoEntry;
  bool do_filter_speckles_;
};

}  // namespace morefusion_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_OCTOMAPSERVER_H_

// Copyright (c) 2019 Kentaro Wada

#include "ros_objslampp_ycb_video/OctomapServer.h"
#include "ros_objslampp_ycb_video/utils.h"
#include <ros_objslampp_srvs/RenderVoxelGridArray.h>

using octomap_msgs::Octomap;

namespace ros_objslampp_ycb_video {

OctomapServer::OctomapServer(ros::NodeHandle private_nh_)
: m_nh(),
  m_octrees(),
  m_instanceCounter(0),
  m_classIds(),
  m_centers(),
  m_pointCloudSub(NULL),
  m_labelInsSub(NULL),
  m_maxRange(-1.0),
  m_worldFrameId("/map"),
  m_sensorFrameId("camera_color_optical_frame"),
  m_res(0.05),
  m_probHit(0.7),
  m_probMiss(0.4),
  m_thresMin(0.12),
  m_thresMax(0.97),
  m_treeDepth(16),
  m_maxTreeDepth(16),
  m_occupancyMinZ(-std::numeric_limits<double>::max()),
  m_occupancyMaxZ(std::numeric_limits<double>::max()),
  m_minSizeX(0.0), m_minSizeY(0.0),
  m_filterSpeckles(false),
  m_compressMap(true) {
  double probHit, probMiss, thresMin, thresMax;

  ros::NodeHandle private_nh(private_nh_);
  private_nh.param("frame_id", m_worldFrameId, m_worldFrameId);
  private_nh.param("sensor_frame_id", m_sensorFrameId, m_sensorFrameId);

  private_nh.param("occupancy_min_z", m_occupancyMinZ, m_occupancyMinZ);
  private_nh.param("occupancy_max_z", m_occupancyMaxZ, m_occupancyMaxZ);
  private_nh.param("min_x_size", m_minSizeX, m_minSizeX);
  private_nh.param("min_y_size", m_minSizeY, m_minSizeY);
  private_nh.param("filter_speckles", m_filterSpeckles, m_filterSpeckles);

  private_nh.param("sensor_model/max_range", m_maxRange, m_maxRange);

  private_nh.param("resolution", m_res, m_res);
  private_nh.param("sensor_model/hit", m_probHit, m_probHit);
  private_nh.param("sensor_model/miss", m_probMiss, m_probMiss);
  private_nh.param("sensor_model/min", m_thresMin, m_thresMin);
  private_nh.param("sensor_model/max", m_thresMax, m_thresMax);
  private_nh.param("compress_map", m_compressMap, m_compressMap);

  m_binaryMapPub = m_nh.advertise<Octomap>("octomap_binary", 1);
  m_fullMapPub = m_nh.advertise<Octomap>("octomap_full", 1);
  m_mapPub = m_nh.advertise<nav_msgs::OccupancyGrid>("projected_map", 5);
  m_fmarkerPub = m_nh.advertise<visualization_msgs::MarkerArray>("free_cells_vis_array", 1);
  m_gridsForRenderPub = private_nh.advertise<ros_objslampp_msgs::VoxelGridArray>("output/grids_for_render", 1);
  m_gridsPub = private_nh.advertise<ros_objslampp_msgs::VoxelGridArray>("output/grids", 1);
  m_gridsNoEntryPub = private_nh.advertise<ros_objslampp_msgs::VoxelGridArray>("output/grids_noentry", 1);
  m_bgMarkerPub = private_nh.advertise<visualization_msgs::MarkerArray>("output/markers_bg", 1);
  m_fgMarkerPub = private_nh.advertise<visualization_msgs::MarkerArray>("output/markers_fg", 1);
  m_labelRenderedPub = private_nh.advertise<sensor_msgs::Image>("output/label_rendered", 1);
  m_labelTrackedPub = private_nh.advertise<sensor_msgs::Image>("output/label_tracked", 1);
  m_classPub = private_nh.advertise<ros_objslampp_msgs::ObjectClassArray>("output/class", 1);

  m_camSub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(
    private_nh, "input/camera_info", 5);
  m_depthSub = new message_filters::Subscriber<sensor_msgs::Image>(
    private_nh, "input/depth", 5);
  m_pointCloudSub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(
    private_nh, "input/points", 5);
  m_labelInsSub = new message_filters::Subscriber<sensor_msgs::Image>(
    private_nh, "input/label_ins", 5);
  m_classSub = new message_filters::Subscriber<ros_objslampp_msgs::ObjectClassArray>(
    private_nh, "input/class", 5);
  m_sync = new message_filters::Synchronizer<ExactSyncPolicy>(100);
  m_sync->connectInput(*m_camSub, *m_depthSub, *m_pointCloudSub, *m_labelInsSub, *m_classSub);
  m_sync->registerCallback(boost::bind(&OctomapServer::insertCloudCallback, this, _1, _2, _3, _4, _5));

  m_renderClient = private_nh.serviceClient<ros_objslampp_srvs::RenderVoxelGridArray>("render");

  dynamic_reconfigure::Server<ros_objslampp_ycb_video::OctomapServerConfig>::CallbackType f =
    boost::bind(&OctomapServer::configCallback, this, _1, _2);
  m_reconfigSrv.setCallback(f);

  ROS_INFO_BLUE("Initialized");
}

void OctomapServer::configCallback(
  const ros_objslampp_ycb_video::OctomapServerConfig& config, const uint32_t level) {
  ROS_INFO_BLUE("configCallback");
  m_groundAsNoEntry = config.ground_as_noentry;
  m_freeAsNoEntry = config.free_as_noentry;
}

void OctomapServer::insertCloudCallback(
    const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
    const sensor_msgs::ImageConstPtr& depth_msg,
    const sensor_msgs::PointCloud2ConstPtr& cloud,
    const sensor_msgs::ImageConstPtr& ins_msg,
    const ros_objslampp_msgs::ObjectClassArrayConstPtr& class_msg) {
  // Get TF
  tf::StampedTransform sensorToWorldTf;
  try {
    m_tfListener.lookupTransform(
      m_worldFrameId, cloud->header.frame_id, cloud->header.stamp, sensorToWorldTf);
  } catch (tf::TransformException& ex) {
    ROS_ERROR_STREAM("Transform error of sensor data: " << ex.what() << ", quitting callback");
    return;
  }
  Eigen::Matrix4f sensorToWorld;
  pcl_ros::transformAsMatrix(sensorToWorldTf, sensorToWorld);

  ros_objslampp_srvs::RenderVoxelGridArray srv;
  tf::transformStampedTFToMsg(sensorToWorldTf, srv.request.transform);
  srv.request.camera_info = *camera_info_msg;
  srv.request.depth = *depth_msg;
  getGridsInWorldFrame(camera_info_msg->header.stamp, srv.request.grids);
  m_renderClient.call(srv);
  m_labelRenderedPub.publish(srv.response.label_ins);

  sensor_msgs::Image ins_rendered_msg = srv.response.label_ins;

  // ROSMsg -> PCL
  PCLPointCloud pc;
  pcl::fromROSMsg(*cloud, pc);

  // ROSMsg -> OpenCV
  cv::Mat label_ins = cv_bridge::toCvCopy(ins_msg, ins_msg->encoding)->image;
  cv::Mat label_ins_rend = cv_bridge::toCvCopy(
    srv.response.label_ins, srv.response.label_ins.encoding)->image;

  // TODO(wkentaro): compute centroid and size of each instance,
  // and remove some of them from integration

  // Transform pointcloud: sensor -> world (map)
  pcl::transformPointCloud(pc, pc, sensorToWorld);

  // Track Instance IDs
  std::map<int, unsigned> instance_id_to_class_id;
  for (size_t i = 0; i < class_msg->classes.size(); i++) {
    instance_id_to_class_id.insert(
      std::make_pair(
        class_msg->classes[i].instance_id,
        class_msg->classes[i].class_id));
  }
  ros_objslampp_ycb_video::utils::track_instance_id(
    /*reference=*/label_ins_rend,
    /*target=*/&label_ins,
    /*instance_id_to_class_id=*/&instance_id_to_class_id,
    /*instance_counter=*/&m_instanceCounter);
  for (std::map<int, unsigned>::iterator it = m_classIds.begin();
       it != m_classIds.end(); it++) {
    if (instance_id_to_class_id.find(it->first) == instance_id_to_class_id.end()) {
      instance_id_to_class_id.insert(std::make_pair(it->first, it->second));
    }
  }
  // Publish Tracked Instance Label
  m_labelTrackedPub.publish(
    cv_bridge::CvImage(ins_msg->header, "32SC1", label_ins).toImageMsg());

  ros_objslampp_msgs::ObjectClassArray cls_rend_msg;
  cls_rend_msg.header = cloud->header;
  for (std::map<int, unsigned>::iterator it = m_classIds.begin();
       it != m_classIds.end(); it++) {
    if (it->first == -1) {
      continue;
    }
    ros_objslampp_msgs::ObjectClass cls;
    cls.instance_id = it->first;
    cls.class_id = it->second;
    cls.confidence = 1;
    cls_rend_msg.classes.push_back(cls);
  }
  m_classPub.publish(cls_rend_msg);

  // Update Map
  insertScan(sensorToWorldTf.getOrigin(), pc, label_ins, instance_id_to_class_id);

  // Publish Object Grids
  publishGrids(cloud->header.stamp, sensorToWorld);

  // Publish Map
  publishAll(cloud->header.stamp);
}

void OctomapServer::insertScan(
    const tf::Point& sensorOriginTf,
    const PCLPointCloud& pc,
    const cv::Mat& label_ins,
    const std::map<int, unsigned>& instance_id_to_class_id) {
  // ros::WallTime t_start = ros::WallTime::now();

  octomap::point3d sensorOrigin = octomap::pointTfToOctomap(sensorOriginTf);

  std::vector<int> instance_ids = ros_objslampp_ycb_video::utils::unique<int>(label_ins);
  octomap::KeySet free_cells_bg;
  std::map<int, octomap::KeySet> occupied_cells;
  for (size_t i = 0; i < instance_ids.size(); i++) {
    int instance_id = instance_ids[i];
    if (instance_id == -2) {
      // -1: background, -2: uncertain (e.g., boundary)
      continue;
    }
    unsigned class_id = 0;
    double pitch = m_res;
    if (instance_id >= 0) {
      if (instance_id_to_class_id.find(instance_id) == instance_id_to_class_id.end()) {
        ROS_FATAL("Can't find instance_id [%d] in instance_id_to_class_id", instance_id);
        return;
      } else {
        class_id = instance_id_to_class_id.find(instance_id)->second;
      }
      pitch = ros_objslampp_ycb_video::utils::class_id_to_voxel_pitch(class_id);
    }
    if (m_octrees.find(instance_id) == m_octrees.end()) {
      OcTreeT* octree = new OcTreeT(pitch);
      octree->setProbHit(m_probHit);
      octree->setProbMiss(m_probMiss);
      octree->setClampingThresMin(m_thresMin);
      octree->setClampingThresMax(m_thresMax);
      m_octrees.insert(std::make_pair(instance_id, octree));
      m_classIds.insert(std::make_pair(instance_id, class_id));
    }
    occupied_cells.insert(std::make_pair(instance_id, octomap::KeySet()));
  }

  // all other points: free on ray, occupied on endpoint:
  std::map<int, PCLPointCloud> instance_id_to_points;
  #pragma omp parallel for
  for (size_t index = 0 ; index < pc.points.size(); index++) {
    size_t width_index = index % pc.width;
    size_t height_index = index / pc.width;
    if (width_index % 2 != 0 || height_index % 2 != 0) {
      continue;
    }
    if (std::isnan(pc.points[index].x) ||
        std::isnan(pc.points[index].y) ||
        std::isnan(pc.points[index].z)) {
      continue;
    }

    octomap::point3d point(pc.points[index].x, pc.points[index].y, pc.points[index].z);
    int instance_id = label_ins.at<int32_t>(height_index, width_index);
    if (instance_id == -2) {
      continue;
    }
    OcTreeT* octree = m_octrees.find(instance_id)->second;

    #pragma omp critical
    {
      if (instance_id_to_points.find(instance_id) == instance_id_to_points.end()) {
        instance_id_to_points.insert(std::make_pair(instance_id, PCLPointCloud()));
      } else {
        instance_id_to_points.find(instance_id)->second.push_back(pc.points[index]);
      }
    }

    // maxrange check
    if ((m_maxRange < 0.0) || ((point - sensorOrigin).norm() <= m_maxRange)) {
      // free cells
      octomap::KeyRay key_ray;
      OcTreeT* octree_bg = m_octrees.find(-1)->second;
      if (octree_bg->computeRayKeys(sensorOrigin, point, key_ray)) {
        #pragma omp critical
        free_cells_bg.insert(key_ray.begin(), key_ray.end());
      }
      // occupied endpoint
      octomap::OcTreeKey key;
      if (m_octrees.find(instance_id)->second->coordToKeyChecked(point, key)) {
        #pragma omp critical
        occupied_cells.find(instance_id)->second.insert(key);
      }
    } else {  // ray longer than maxrange:;
      octomap::point3d new_end = sensorOrigin + (point - sensorOrigin).normalized() * m_maxRange;
      octomap::KeyRay key_ray;
      OcTreeT* octree_bg = m_octrees.find(-1)->second;
      if (octree_bg->computeRayKeys(sensorOrigin, new_end, key_ray)) {
        #pragma omp critical
        free_cells_bg.insert(key_ray.begin(), key_ray.end());
      }
    }
  }

  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  octomap::KeySet occupied_cells_bg = occupied_cells.find(-1)->second;
  for (octomap::KeySet::iterator it = free_cells_bg.begin(); it != free_cells_bg.end(); it++) {
    if (occupied_cells_bg.find(*it) == occupied_cells_bg.end()) {
      octree_bg->updateNode(*it, false);
    }
  }

  for (std::map<int, octomap::KeySet>::iterator i = occupied_cells.begin(); i != occupied_cells.end(); i++) {
    int instance_id = i->first;
    octomap::KeySet key_set_occupied = i->second;
    OcTreeT* octree = m_octrees.find(instance_id)->second;
    for (octomap::KeySet::iterator j = key_set_occupied.begin(); j != key_set_occupied.end(); j++) {
      octree->updateNode(*j, true);
    }
  }

  for (std::map<int, PCLPointCloud>::iterator it = instance_id_to_points.begin();
       it != instance_id_to_points.end(); it++) {
    int instance_id = it->first;
    Eigen::Matrix<float, 4, 1> centroid;
    pcl::compute3DCentroid<PCLPoint, float>(
      /*cloud=*/it->second, /*centroid=*/centroid);
    octomap::point3d center(centroid(0, 0), centroid(1, 0), centroid(2, 0));
    m_centers.insert(std::make_pair(instance_id, center));
  }

  if (m_compressMap) {
    for (std::map<int, OcTreeT*>::iterator it = m_octrees.begin(); it != m_octrees.end(); it++) {
      it->second->prune();
    }
  }

  // ros::WallDuration elapsed_time = ros::WallTime::now() - t_start;
  // ROS_INFO_MAGENTA("Elapsed Time: %lf [s], %lf [fps]", elapsed_time.toSec(), 1. / elapsed_time.toSec());
}

void OctomapServer::getGridsInWorldFrame(const ros::Time& rostime, ros_objslampp_msgs::VoxelGridArray& grids) {
  grids.header.frame_id = m_worldFrameId;
  grids.header.stamp = rostime;
  for (std::map<int, OcTreeT*>::iterator it_octree = m_octrees.begin();
       it_octree != m_octrees.end(); it_octree++) {
    int instance_id = it_octree->first;
    OcTreeT* octree = it_octree->second;

    if (instance_id == -1) {
      continue;
    }
    unsigned class_id = m_classIds.find(instance_id)->second;
    double pitch = ros_objslampp_ycb_video::utils::class_id_to_voxel_pitch(class_id);

    // world frame
    octomap::point3d center = m_centers.find(instance_id)->second;

    ros_objslampp_msgs::VoxelGrid grid;
    grid.pitch = pitch;
    grid.dims.x = 32;
    grid.dims.y = 32;
    grid.dims.z = 32;
    grid.origin.x = center.x() - (grid.dims.x / 2.0 - 0.5) * grid.pitch;
    grid.origin.y = center.y() - (grid.dims.y / 2.0 - 0.5) * grid.pitch;
    grid.origin.z = center.z() - (grid.dims.z / 2.0 - 0.5) * grid.pitch;
    grid.instance_id = instance_id;
    grid.class_id = class_id;

    for (size_t i = 0; i < grid.dims.x; i++) {
      for (size_t j = 0; j < grid.dims.y; j++) {
        for (size_t k = 0; k < grid.dims.z; k++) {
          double x, y, z;
          // in world
          x = grid.origin.x + grid.pitch * i;
          y = grid.origin.y + grid.pitch * j;
          z = grid.origin.z + grid.pitch * k;

          size_t index = i * grid.dims.y * grid.dims.z + j * grid.dims.z + k;

          octomap::OcTreeNode* node = octree->search(x, y, z, /*depth=*/0);
          if ((node != NULL) && (node->getOccupancy() > 0.5)) {
            grid.indices.push_back(index);
            grid.values.push_back(node->getOccupancy());
          }
        }
      }
    }
    grids.grids.push_back(grid);
  }
}

void OctomapServer::publishGrids(
    const ros::Time& rostime,
    const Eigen::Matrix4f& sensorToWorld) {
  if (m_octrees.size() == 0) {
    return;
  }

  ros_objslampp_msgs::VoxelGridArray grids;
  grids.header.frame_id = m_sensorFrameId;
  grids.header.stamp = rostime;
  ros_objslampp_msgs::VoxelGridArray grids_noentry;
  grids_noentry.header = grids.header;
  for (std::map<int, OcTreeT*>::iterator it_octree = m_octrees.begin();
       it_octree != m_octrees.end(); it_octree++) {
    int instance_id = it_octree->first;
    OcTreeT* octree = it_octree->second;

    if (instance_id == -1) {
      continue;
    }
    unsigned class_id = m_classIds.find(instance_id)->second;
    double pitch = ros_objslampp_ycb_video::utils::class_id_to_voxel_pitch(class_id);

    octomap::point3d center = m_centers.find(instance_id)->second;

    PCLPointCloud center_sensor;
    center_sensor.push_back(PCLPoint(center.x(), center.y(), center.z()));
    pcl::transformPointCloud(center_sensor, center_sensor, sensorToWorld.inverse());

    ros_objslampp_msgs::VoxelGrid grid;
    grid.pitch = pitch;
    grid.dims.x = 32;
    grid.dims.y = 32;
    grid.dims.z = 32;
    grid.origin.x = center_sensor.points[0].x - (grid.dims.x / 2.0 - 0.5) * grid.pitch;
    grid.origin.y = center_sensor.points[0].y - (grid.dims.y / 2.0 - 0.5) * grid.pitch;
    grid.origin.z = center_sensor.points[0].z - (grid.dims.z / 2.0 - 0.5) * grid.pitch;
    grid.instance_id = instance_id;
    grid.class_id = class_id;

    ros_objslampp_msgs::VoxelGrid grid_noentry;
    grid_noentry.pitch = grid.pitch;
    grid_noentry.dims = grid.dims;
    grid_noentry.origin = grid.origin;
    grid_noentry.instance_id = grid.instance_id;
    grid_noentry.class_id = grid.class_id;
    for (size_t i = 0; i < grid.dims.x; i++) {
      for (size_t j = 0; j < grid.dims.y; j++) {
        for (size_t k = 0; k < grid.dims.z; k++) {
          double x, y, z;
          // in sensor
          x = grid.origin.x + grid.pitch * i;
          y = grid.origin.y + grid.pitch * j;
          z = grid.origin.z + grid.pitch * k;

          // in world
          PCLPointCloud p_world;
          p_world.push_back(PCLPoint(x, y, z));  // sensor
          pcl::transformPointCloud(p_world, p_world, sensorToWorld);
          x = p_world.points[0].x;
          y = p_world.points[0].y;
          z = p_world.points[0].z;

          size_t index = i * grid.dims.y * grid.dims.z + j * grid.dims.z + k;
          if (m_groundAsNoEntry && (z < 0)) {
            grid_noentry.indices.push_back(index);
            grid_noentry.values.push_back(m_thresMin);
            continue;
          }

          octomap::OcTreeNode* node = octree->search(x, y, z, /*depth=*/0);
          if ((node != NULL) && (node->getOccupancy() > 0.5)) {
            grid.indices.push_back(index);
            grid.values.push_back(node->getOccupancy());
          } else {
            for (std::map<int, OcTreeT*>::iterator it_octree_other = m_octrees.begin();
                 it_octree_other != m_octrees.end(); it_octree_other++) {
              if (it_octree_other->first == instance_id) {
                continue;
              }
              OcTreeT* octree_other = it_octree_other->second;
              node = octree_other->search(x, y, z, /*depth=*/0);
              if (node != NULL) {
                double occupancy = node->getOccupancy();
                if ((it_octree_other->first == -1) &&
                    m_freeAsNoEntry && (occupancy < 0.5)) {
                  grid_noentry.indices.push_back(index);
                  grid_noentry.values.push_back(1 - occupancy);
                } else if (occupancy >= m_thresMax) {
                  grid_noentry.indices.push_back(index);
                  grid_noentry.values.push_back(occupancy);
                }
              }
            }
          }
        }
      }
    }
    grids.grids.push_back(grid);
    grids_noentry.grids.push_back(grid_noentry);
  }
  m_gridsPub.publish(grids);
  m_gridsNoEntryPub.publish(grids_noentry);
}

void OctomapServer::publishAll(const ros::Time& rostime) {
  if (m_octrees.size() == 0) {
    return;
  }

  bool publishFreeMarkerArray = m_fmarkerPub.getNumSubscribers() > 0;
  bool publishMarkerArray = m_bgMarkerPub.getNumSubscribers() > 0 || m_fgMarkerPub.getNumSubscribers() > 0;
  bool publishBinaryMap = m_binaryMapPub.getNumSubscribers() > 0;
  bool publishFullMap = m_fullMapPub.getNumSubscribers() > 0;

  // init markers for free space:
  visualization_msgs::MarkerArray freeNodesVis;
  // each array stores all cubes of a different size, one for each depth level:
  freeNodesVis.markers.resize(m_treeDepth+1);

  geometry_msgs::Pose pose;
  pose.orientation = tf::createQuaternionMsgFromYaw(0.0);

  // now, traverse all leafs in the tree:
  std::map<int, visualization_msgs::MarkerArray> occupiedNodesVisAll;
  for (std::map<int, OcTreeT*>::iterator it_octree = m_octrees.begin();
       it_octree != m_octrees.end(); it_octree++) {
    // init markers:
    visualization_msgs::MarkerArray occupiedNodesVis;
    // each array stores all cubes of a different size, one for each depth level:
    occupiedNodesVis.markers.resize(m_treeDepth+1);

    const int instance_id = it_octree->first;
    OcTreeT* octree = it_octree->second;
    for (OcTreeT::iterator it = octree->begin(m_maxTreeDepth);
         it != octree->end(); it++) {
      if (octree->isNodeOccupied(*it)) {
        double z = it.getZ();
        double half_size = it.getSize() / 2.0;
        if (z + half_size > m_occupancyMinZ && z - half_size < m_occupancyMaxZ) {
          double size = it.getSize();
          double x = it.getX();
          double y = it.getY();

          // Ignore speckles in the map:
          if (m_filterSpeckles && (it.getDepth() == m_treeDepth +1) && isSpeckleNode(it.getKey())) {
            ROS_DEBUG("Ignoring single speckle at (%f,%f,%f)", x, y, z);
            continue;
          }  // else: current octree node is no speckle, send it out

          // create marker:
          if (publishMarkerArray) {
            unsigned idx = it.getDepth();
            assert(idx < occupiedNodesVis.markers.size());

            geometry_msgs::Point cubeCenter;
            cubeCenter.x = x;
            cubeCenter.y = y;
            cubeCenter.z = z;

            occupiedNodesVis.markers[idx].points.push_back(cubeCenter);
          }
        }
      } else if (instance_id != -1) {
        continue;
      } else {
        // node not occupied => mark as free in 2D map if unknown so far
        double z = it.getZ();
        double half_size = it.getSize() / 2.0;
        if (z + half_size > m_occupancyMinZ && z - half_size < m_occupancyMaxZ) {
          if (publishFreeMarkerArray) {
            double x = it.getX();
            double y = it.getY();

            // create marker for free space:
            unsigned idx = it.getDepth();
            assert(idx < freeNodesVis.markers.size());

            geometry_msgs::Point cubeCenter;
            cubeCenter.x = x;
            cubeCenter.y = y;
            cubeCenter.z = z;

            freeNodesVis.markers[idx].points.push_back(cubeCenter);
          }
        }
      }
    }

    // finish MarkerArray:
    if (publishMarkerArray) {
      for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i) {
        double size = octree->getNodeSize(i);

        occupiedNodesVis.markers[i].header.frame_id = m_worldFrameId;
        occupiedNodesVis.markers[i].header.stamp = rostime;
        occupiedNodesVis.markers[i].ns = boost::lexical_cast<std::string>(instance_id);
        occupiedNodesVis.markers[i].id = i;
        occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
        occupiedNodesVis.markers[i].scale.x = size;
        occupiedNodesVis.markers[i].scale.y = size;
        occupiedNodesVis.markers[i].scale.z = size;
        occupiedNodesVis.markers[i].color =
          ros_objslampp_ycb_video::utils::colorCategory40(instance_id + 1);

        if (occupiedNodesVis.markers[i].points.size() > 0)
          occupiedNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
        else
          occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
      }
    }
    occupiedNodesVisAll.insert(std::make_pair(instance_id, occupiedNodesVis));
  }

  if (publishMarkerArray) {
    visualization_msgs::MarkerArray occupiedNodesVisBG;
    visualization_msgs::MarkerArray occupiedNodesVisFG;
    for (std::map<int, visualization_msgs::MarkerArray>::iterator it = occupiedNodesVisAll.begin();
         it != occupiedNodesVisAll.end(); it++) {
      const int instance_id = it->first;
      for (size_t j = 0; j < it->second.markers.size(); j++) {
        if (instance_id == -1) {
          occupiedNodesVisBG.markers.push_back(it->second.markers[j]);
        } else {
          occupiedNodesVisFG.markers.push_back(it->second.markers[j]);
        }
      }
    }
    m_bgMarkerPub.publish(occupiedNodesVisBG);
    m_fgMarkerPub.publish(occupiedNodesVisFG);
  }

  // finish FreeMarkerArray:
  if (publishFreeMarkerArray) {
    OcTreeT* octree_bg = m_octrees.find(-1)->second;
    for (unsigned i= 0; i < freeNodesVis.markers.size(); ++i) {
      double size = octree_bg->getNodeSize(i);

      freeNodesVis.markers[i].header.frame_id = m_worldFrameId;
      freeNodesVis.markers[i].header.stamp = rostime;
      freeNodesVis.markers[i].ns = "map";
      freeNodesVis.markers[i].id = i;
      freeNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
      freeNodesVis.markers[i].scale.x = size;
      freeNodesVis.markers[i].scale.y = size;
      freeNodesVis.markers[i].scale.z = size;
      freeNodesVis.markers[i].color.r = 0.5;
      freeNodesVis.markers[i].color.g = 0.5;
      freeNodesVis.markers[i].color.b = 0.5;
      freeNodesVis.markers[i].color.a = 1.0;


      if (freeNodesVis.markers[i].points.size() > 0) {
        freeNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
      } else {
        freeNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
      }
    }

    m_fmarkerPub.publish(freeNodesVis);
  }

  if (publishBinaryMap) {
    publishBinaryOctoMap(rostime);
  }

  if (publishFullMap) {
    publishFullOctoMap(rostime);
  }
}


void OctomapServer::publishBinaryOctoMap(const ros::Time& rostime) const {
  Octomap map;
  map.header.frame_id = m_worldFrameId;
  map.header.stamp = rostime;

  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  if (octomap_msgs::binaryMapToMsg(*octree_bg, map)) {
    m_binaryMapPub.publish(map);
  } else {
    ROS_ERROR("Error serializing OctoMap");
  }
}

void OctomapServer::publishFullOctoMap(const ros::Time& rostime) const {
  Octomap map;
  map.header.frame_id = m_worldFrameId;
  map.header.stamp = rostime;

  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  if (octomap_msgs::fullMapToMsg(*octree_bg, map)) {
    m_fullMapPub.publish(map);
  } else {
    ROS_ERROR("Error serializing OctoMap");
  }
}

bool OctomapServer::isSpeckleNode(const octomap::OcTreeKey& nKey) const {
  octomap::OcTreeKey key;
  bool neighborFound = false;
  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  for (key[2] = nKey[2] - 1; !neighborFound && key[2] <= nKey[2] + 1; ++key[2]) {
    for (key[1] = nKey[1] - 1; !neighborFound && key[1] <= nKey[1] + 1; ++key[1]) {
      for (key[0] = nKey[0] - 1; !neighborFound && key[0] <= nKey[0] + 1; ++key[0]) {
        if (key != nKey) {
          octomap::OcTreeNode* node = octree_bg->search(key);
          if (node && octree_bg->isNodeOccupied(node)) {
            // we have a neighbor => break!
            neighborFound = true;
          }
        }
      }
    }
  }

  return neighborFound;
}

}  // namespace ros_objslampp_ycb_video

int main(int argc, char** argv) {
  ros::init(argc, argv, "octomap_server");
  const ros::NodeHandle& private_nh = ros::NodeHandle("~");

  ros_objslampp_ycb_video::OctomapServer server;

  try {
    ros::spin();
  } catch (std::runtime_error& e) {
    ROS_ERROR("octomap_server exception: %s", e.what());
    return -1;
  }

  return 0;
}

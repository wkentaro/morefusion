#include "ros_objslampp_ycb_video/color_utils.h"
#include "ros_objslampp_ycb_video/OctomapServer.h"
#include "ros_objslampp_ycb_video/log_utils.h"
#include <ros_objslampp_msgs/VoxelGridArray.h>

using namespace octomap;
using octomap_msgs::Octomap;

bool is_equal (double a, double b, double epsilon = 1.0e-7)
{
    return std::abs(a - b) < epsilon;
}

namespace ros_objslampp_ycb_video {

OctomapServer::OctomapServer(ros::NodeHandle private_nh_)
: m_nh(),
  m_octrees(),
  m_pointCloudSub(NULL),
  m_labelInsSub(NULL),
  m_tfPointCloudSub(NULL),
  m_maxRange(-1.0),
  m_worldFrameId("/map"),
  m_baseFrameId("base_footprint"),
  m_latchedTopics(true),
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
  m_compressMap(true),
  m_stopUpdate(false)
{
  double probHit, probMiss, thresMin, thresMax;

  ros::NodeHandle private_nh(private_nh_);
  private_nh.param("frame_id", m_worldFrameId, m_worldFrameId);
  private_nh.param("base_frame_id", m_baseFrameId, m_baseFrameId);

  private_nh.param("occupancy_min_z", m_occupancyMinZ,m_occupancyMinZ);
  private_nh.param("occupancy_max_z", m_occupancyMaxZ,m_occupancyMaxZ);
  private_nh.param("min_x_size", m_minSizeX,m_minSizeX);
  private_nh.param("min_y_size", m_minSizeY,m_minSizeY);
  private_nh.param("filter_speckles", m_filterSpeckles, m_filterSpeckles);

  private_nh.param("sensor_model/max_range", m_maxRange, m_maxRange);

  private_nh.param("resolution", m_res, m_res);
  private_nh.param("sensor_model/hit", m_probHit, m_probHit);
  private_nh.param("sensor_model/miss", m_probMiss, m_probMiss);
  private_nh.param("sensor_model/min", m_thresMin, m_thresMin);
  private_nh.param("sensor_model/max", m_thresMax, m_thresMax);
  private_nh.param("compress_map", m_compressMap, m_compressMap);

  private_nh.param("latch", m_latchedTopics, m_latchedTopics);
  if (m_latchedTopics){
    ROS_INFO("Publishing latched (single publish will take longer, all topics are prepared)");
  } else
    ROS_INFO("Publishing non-latched (topics are only prepared as needed, will only be re-published on map change");

  m_markerPub = m_nh.advertise<visualization_msgs::MarkerArray>("occupied_cells_vis_array", 1, m_latchedTopics);
  m_binaryMapPub = m_nh.advertise<Octomap>("octomap_binary", 1, m_latchedTopics);
  m_fullMapPub = m_nh.advertise<Octomap>("octomap_full", 1, m_latchedTopics);
  m_pointCloudPub = m_nh.advertise<sensor_msgs::PointCloud2>("octomap_point_cloud_centers", 1, m_latchedTopics);
  m_mapPub = m_nh.advertise<nav_msgs::OccupancyGrid>("projected_map", 5, m_latchedTopics);
  m_fmarkerPub = m_nh.advertise<visualization_msgs::MarkerArray>("free_cells_vis_array", 1, m_latchedTopics);
  m_bboxesPub = private_nh.advertise<jsk_recognition_msgs::BoundingBoxArray>(
    "output/bboxes", 1, m_latchedTopics);
  m_gridsPub = private_nh.advertise<ros_objslampp_msgs::VoxelGridArray>(
    "output/grids", 1, m_latchedTopics);
  m_gridsNoEntryPub = private_nh.advertise<ros_objslampp_msgs::VoxelGridArray>(
    "output/grids_noentry", 1, m_latchedTopics);

  m_pointCloudSub = new message_filters::Subscriber<sensor_msgs::PointCloud2> (m_nh, "cloud_in", 5);
  m_labelInsSub = new message_filters::Subscriber<sensor_msgs::Image> (m_nh, "label_ins_in", 5);
  m_sync = new message_filters::Synchronizer<ExactSyncPolicy>(100);
  m_sync->connectInput(*m_pointCloudSub, *m_labelInsSub);
  m_sync->registerCallback(boost::bind(&OctomapServer::insertCloudCallback, this, _1, _2));

  m_octomapBinaryService = m_nh.advertiseService("octomap_binary", &OctomapServer::octomapBinarySrv, this);
  m_octomapFullService = m_nh.advertiseService("octomap_full", &OctomapServer::octomapFullSrv, this);
  m_clearBBXService = private_nh.advertiseService("clear_bbx", &OctomapServer::clearBBXSrv, this);
  m_resetService = private_nh.advertiseService("reset", &OctomapServer::resetSrv, this);

  ROS_INFO_BLUE("Initialized");
}

OctomapServer::~OctomapServer(){
  if (m_tfPointCloudSub){
    delete m_tfPointCloudSub;
    m_tfPointCloudSub = NULL;
  }

  if (m_pointCloudSub){
    delete m_pointCloudSub;
    m_pointCloudSub = NULL;
  }

  if (m_labelInsSub){
    delete m_labelInsSub;
    m_labelInsSub = NULL;
  }

  if (m_octrees.size()) {
    delete &m_octrees;
    m_octrees.clear();
  }
}

void OctomapServer::insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud, const sensor_msgs::ImageConstPtr& ins_msg) {
  ROS_INFO_BLUE("insertCloudCallback");

  PCLPointCloud pc;
  pcl::fromROSMsg(*cloud, pc);

  tf::StampedTransform sensorToWorldTf;
  try {
    m_tfListener.lookupTransform(m_worldFrameId, cloud->header.frame_id, cloud->header.stamp, sensorToWorldTf);
  } catch(tf::TransformException& ex){
    ROS_ERROR_STREAM( "Transform error of sensor data: " << ex.what() << ", quitting callback");
    return;
  }

  Eigen::Matrix4f sensorToWorld;
  pcl_ros::transformAsMatrix(sensorToWorldTf, sensorToWorld);
  pcl::transformPointCloud(pc, pc, sensorToWorld);

  cv::Mat label_ins = cv_bridge::toCvCopy(ins_msg, ins_msg->encoding)->image;
  if (!((cloud->height == ins_msg->height) && (cloud->width == ins_msg->width))) {
    ROS_ERROR("Point cloud and instance label must be same size!");
    ROS_ERROR("point cloud: (%d, %d), label instance: (%d, %d)",
              cloud->height, cloud->width, ins_msg->height, ins_msg->width);
    return;
  }

  if (!m_stopUpdate) {
    insertScan(sensorToWorldTf.getOrigin(), pc, label_ins);
  }

  publishAll(cloud->header.stamp);
}

void OctomapServer::insertScan(
  const tf::Point& sensorOriginTf,
  const PCLPointCloud& pc,
  const cv::Mat& label_ins)
{
  point3d sensorOrigin = pointTfToOctomap(sensorOriginTf);

  // all other points: free on ray, occupied on endpoint:
  KeySet free_cells, occupied_cells;
  for (size_t index = 0 ; index < pc.points.size(); index++)
  {
    size_t width_index = index % pc.width;
    size_t height_index = index / pc.width;
    if (isnan(pc.points[index].x) || isnan(pc.points[index].y) || isnan(pc.points[index].z))
    {
      // Skip NaN points
      continue;
    }

    octomap::point3d point(pc.points[index].x, pc.points[index].y, pc.points[index].z);
    int instance_id = label_ins.at<uint32_t>(height_index, width_index);
    if (m_octrees.find(instance_id) == m_octrees.end())
    {
      OcTreeT* octree = new OcTreeT(m_res);
      octree->setProbHit(m_probHit);
      octree->setProbMiss(m_probMiss);
      octree->setClampingThresMin(m_thresMin);
      octree->setClampingThresMax(m_thresMax);
      m_octrees.insert(std::make_pair(instance_id, octree));
    }
    OcTreeT* octree = m_octrees.find(instance_id)->second;

    // maxrange check
    if ((m_maxRange < 0.0) || ((point - sensorOrigin).norm() <= m_maxRange))
    {
      // free cells
      if (octree->computeRayKeys(sensorOrigin, point, m_keyRay))
      {
        free_cells.insert(m_keyRay.begin(), m_keyRay.end());
      }
      // occupied endpoint
      OcTreeKey key;
      if (octree->coordToKeyChecked(point, key)){
        occupied_cells.insert(key);
        octree->updateNode(key, true);
        updateMinKey(key, m_updateBBXMin);
        updateMaxKey(key, m_updateBBXMax);
      }
    } else {// ray longer than maxrange:;
      point3d new_end = sensorOrigin + (point - sensorOrigin).normalized() * m_maxRange;
      if (octree->computeRayKeys(sensorOrigin, new_end, m_keyRay)){
        free_cells.insert(m_keyRay.begin(), m_keyRay.end());

        octomap::OcTreeKey endKey;
        if (octree->coordToKeyChecked(new_end, endKey)){
          free_cells.insert(endKey);
          updateMinKey(endKey, m_updateBBXMin);
          updateMaxKey(endKey, m_updateBBXMax);
        } else{
          ROS_ERROR_STREAM("Could not generate Key for endpoint "<<new_end);
        }
      }
    }
  }

  // mark free cells only if not seen occupied in this cloud
  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  for(KeySet::iterator it = free_cells.begin(), end=free_cells.end(); it!= end; ++it){
    if (occupied_cells.find(*it) == occupied_cells.end()){
      octree_bg->updateNode(*it, false);
    }
  }

  // TODO: eval lazy+updateInner vs. proper insertion
  // non-lazy by default (updateInnerOccupancy() too slow for large maps)
  //octree->updateInnerOccupancy();
  // octomap::point3d minPt, maxPt;
  // ROS_DEBUG_STREAM("Bounding box keys (before): " << m_updateBBXMin[0] << " " <<m_updateBBXMin[1] << " " << m_updateBBXMin[2] << " / " <<m_updateBBXMax[0] << " "<<m_updateBBXMax[1] << " "<< m_updateBBXMax[2]);

  // TODO: snap max / min keys to larger voxels by m_maxTreeDepth
//   if (m_maxTreeDepth < 16)
//   {
//      OcTreeKey tmpMin = getIndexKey(m_updateBBXMin, m_maxTreeDepth); // this should give us the first key at depth m_maxTreeDepth that is smaller or equal to m_updateBBXMin (i.e. lower left in 2D grid coordinates)
//      OcTreeKey tmpMax = getIndexKey(m_updateBBXMax, m_maxTreeDepth); // see above, now add something to find upper right
//      tmpMax[0]+= octree->getNodeSize( m_maxTreeDepth ) - 1;
//      tmpMax[1]+= octree->getNodeSize( m_maxTreeDepth ) - 1;
//      tmpMax[2]+= octree->getNodeSize( m_maxTreeDepth ) - 1;
//      m_updateBBXMin = tmpMin;
//      m_updateBBXMax = tmpMax;
//   }

  // TODO: we could also limit the bbx to be within the map bounds here (see publishing check)
  // minPt = octree->keyToCoord(m_updateBBXMin);
  // maxPt = octree->keyToCoord(m_updateBBXMax);
  // ROS_DEBUG_STREAM("Updated area bounding box: "<< minPt << " - "<<maxPt);
  // ROS_DEBUG_STREAM("Bounding box keys (after): " << m_updateBBXMin[0] << " " <<m_updateBBXMin[1] << " " << m_updateBBXMin[2] << " / " <<m_updateBBXMax[0] << " "<<m_updateBBXMax[1] << " "<< m_updateBBXMax[2]);

  if (m_compressMap) {
    for (std::map<int, OcTreeT*>::iterator it = m_octrees.begin(); it != m_octrees.end(); it++)
    {
      it->second->prune();
    }
  }

  ROS_INFO_MAGENTA("Stop updating for single-view dev");
  m_stopUpdate = true;
}



void OctomapServer::publishAll(const ros::Time& rostime){
  if (m_octrees.size() == 0) {
    return;
  }

  ros::WallTime startTime = ros::WallTime::now();

  bool publishFreeMarkerArray = (m_latchedTopics || m_fmarkerPub.getNumSubscribers() > 0);
  bool publishMarkerArray = (m_latchedTopics || m_markerPub.getNumSubscribers() > 0);
  bool publishPointCloud = (m_latchedTopics || m_pointCloudPub.getNumSubscribers() > 0);
  bool publishBinaryMap = (m_latchedTopics || m_binaryMapPub.getNumSubscribers() > 0);
  bool publishFullMap = (m_latchedTopics || m_fullMapPub.getNumSubscribers() > 0);

  // init markers for free space:
  visualization_msgs::MarkerArray freeNodesVis;
  // each array stores all cubes of a different size, one for each depth level:
  freeNodesVis.markers.resize(m_treeDepth+1);

  geometry_msgs::Pose pose;
  pose.orientation = tf::createQuaternionMsgFromYaw(0.0);

  // init pointcloud:
  pcl::PointCloud<PCLPoint> pclCloud;

  jsk_recognition_msgs::BoundingBoxArray bboxes;
  bboxes.header.frame_id = m_worldFrameId;
  bboxes.header.stamp = rostime;
  ros_objslampp_msgs::VoxelGridArray grids;
  grids.header = bboxes.header;
  ros_objslampp_msgs::VoxelGridArray grids_noentry;
  grids_noentry.header = bboxes.header;
  for (std::map<int, OcTreeT*>::iterator it_octree = m_octrees.begin(); it_octree != m_octrees.end(); it_octree++)
  {
    int instance_id = it_octree->first;
    OcTreeT* octree = it_octree->second;

    if (instance_id == -1)
    {
      continue;
    }

    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
    double dim_x, dim_y, dim_z;
    octree->getMetricMax(max_x, max_y, max_z);
    octree->getMetricMin(min_x, min_y, min_z);
    octree->getMetricSize(dim_x, dim_y, dim_z);

    jsk_recognition_msgs::BoundingBox bbox;
    bbox.header.frame_id = m_worldFrameId;
    bbox.header.stamp = rostime;
    bbox.pose.position.x = (min_x + max_x) / 2;
    bbox.pose.position.y = (min_y + max_y) / 2;
    bbox.pose.position.z = (min_z + max_z) / 2;
    bbox.dimensions.x = dim_x;
    bbox.dimensions.y = dim_y;
    bbox.dimensions.z = dim_z;
    bboxes.boxes.push_back(bbox);

    ros_objslampp_msgs::VoxelGrid grid;
    grid.pitch = m_res;
    grid.dims.x = 32;
    grid.dims.y = 32;
    grid.dims.z = 32;
    grid.origin.x = bbox.pose.position.x - (grid.dims.x / 2.0 - 0.5) * grid.pitch;
    grid.origin.y = bbox.pose.position.y - (grid.dims.y / 2.0 - 0.5) * grid.pitch;
    grid.origin.z = bbox.pose.position.z - (grid.dims.z / 2.0 - 0.5) * grid.pitch;
    grid.label = instance_id;

    ros_objslampp_msgs::VoxelGrid grid_noentry;
    grid_noentry.pitch = grid.pitch;
    grid_noentry.dims = grid.dims;
    grid_noentry.origin = grid.origin;
    grid_noentry.label = grid.label;
    for (size_t i = 0; i < grid.dims.x; i++) {
      for (size_t j = 0; j < grid.dims.y; j++) {
        for (size_t k = 0; k < grid.dims.z; k++) {
          double x, y, z;
          x = grid.origin.x + grid.pitch * i;
          y = grid.origin.y + grid.pitch * j;
          z = grid.origin.z + grid.pitch * k;
          octomap::OcTreeNode* node = octree->search(x, y, z, /*depth=*/0);
          size_t index = i * grid.dims.y * grid.dims.z + j * grid.dims.z + k;
          if (node != NULL) {
            grid.indices.push_back(index);
            grid.values.push_back(node->getOccupancy());
          } else {
            for (std::map<int, OcTreeT*>::iterator it_octree_other = m_octrees.begin(); it_octree_other != m_octrees.end(); it_octree_other++)
            {
              if (it_octree_other->first == instance_id)
              {
                continue;
              }
              OcTreeT* octree_other = it_octree_other->second;
              node = octree_other->search(x, y, z, /*depth=*/0);
              if (node != NULL) {
                grid_noentry.indices.push_back(index);
              }
            }
          }
        }
      }
    }
    grids.grids.push_back(grid);
    grids_noentry.grids.push_back(grid_noentry);
  }
  m_bboxesPub.publish(bboxes);
  m_gridsPub.publish(grids);
  m_gridsNoEntryPub.publish(grids_noentry);

  // now, traverse all leafs in the tree:
  std::vector<visualization_msgs::MarkerArray> occupiedNodesVisAll;
  for (std::map<int, OcTreeT*>::iterator it_octree = m_octrees.begin(); it_octree != m_octrees.end(); it_octree++)
  {
    // init markers:
    visualization_msgs::MarkerArray occupiedNodesVis;
    // each array stores all cubes of a different size, one for each depth level:
    occupiedNodesVis.markers.resize(m_treeDepth+1);

    int instance_id = it_octree->first;
    OcTreeT* octree = it_octree->second;
    for (OcTreeT::iterator it = octree->begin(m_maxTreeDepth), end = octree->end(); it != end; ++it)
    {
      if (octree->isNodeOccupied(*it)) {
        double z = it.getZ();
        double half_size = it.getSize() / 2.0;
        if (z + half_size > m_occupancyMinZ && z - half_size < m_occupancyMaxZ)
        {
          double size = it.getSize();
          double x = it.getX();
          double y = it.getY();

          // Ignore speckles in the map:
          if (m_filterSpeckles && (it.getDepth() == m_treeDepth +1) && isSpeckleNode(it.getKey())){
            ROS_DEBUG("Ignoring single speckle at (%f,%f,%f)", x, y, z);
            continue;
          } // else: current octree node is no speckle, send it out

          //create marker:
          if (publishMarkerArray){
            unsigned idx = it.getDepth();
            assert(idx < occupiedNodesVis.markers.size());

            geometry_msgs::Point cubeCenter;
            cubeCenter.x = x;
            cubeCenter.y = y;
            cubeCenter.z = z;

            occupiedNodesVis.markers[idx].points.push_back(cubeCenter);
          }

          // insert into pointcloud:
          if (publishPointCloud) {
            pclCloud.push_back(PCLPoint(x, y, z));
          }

        }
      }
      else if (instance_id != -1)
      {
        continue;
      }
      else
      {
        // node not occupied => mark as free in 2D map if unknown so far
        double z = it.getZ();
        double half_size = it.getSize() / 2.0;
        if (z + half_size > m_occupancyMinZ && z - half_size < m_occupancyMaxZ)
        {
          if (publishFreeMarkerArray){
            double x = it.getX();
            double y = it.getY();

            //create marker for free space:
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
      for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i){
        double size = octree->getNodeSize(i);

        occupiedNodesVis.markers[i].header.frame_id = m_worldFrameId;
        occupiedNodesVis.markers[i].header.stamp = rostime;
        occupiedNodesVis.markers[i].ns = boost::lexical_cast<std::string>(instance_id);
        occupiedNodesVis.markers[i].id = i;
        occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
        occupiedNodesVis.markers[i].scale.x = size;
        occupiedNodesVis.markers[i].scale.y = size;
        occupiedNodesVis.markers[i].scale.z = size;
        occupiedNodesVis.markers[i].color = colorCategory40(instance_id + 1);

        if (occupiedNodesVis.markers[i].points.size() > 0)
          occupiedNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
        else
          occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
      }
    }
    occupiedNodesVisAll.push_back(occupiedNodesVis);
  }

  if (publishMarkerArray)
  {
    visualization_msgs::MarkerArray occupiedNodesVis;
    for (size_t i = 0; i < occupiedNodesVisAll.size(); i++) {
      for (size_t j = 0; j < occupiedNodesVisAll[i].markers.size(); j++) {
        occupiedNodesVis.markers.push_back(occupiedNodesVisAll[i].markers[j]);
      }
    }
    m_markerPub.publish(occupiedNodesVis);
  }

  // finish FreeMarkerArray:
  if (publishFreeMarkerArray){
    OcTreeT* octree_bg = m_octrees.find(-1)->second;
    for (unsigned i= 0; i < freeNodesVis.markers.size(); ++i){
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


      if (freeNodesVis.markers[i].points.size() > 0)
        freeNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
      else
        freeNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
    }

    m_fmarkerPub.publish(freeNodesVis);
  }


  // finish pointcloud:
  if (publishPointCloud){
    sensor_msgs::PointCloud2 cloud;
    pcl::toROSMsg (pclCloud, cloud);
    cloud.header.frame_id = m_worldFrameId;
    cloud.header.stamp = rostime;
    m_pointCloudPub.publish(cloud);
  }

  if (publishBinaryMap)
    publishBinaryOctoMap(rostime);

  if (publishFullMap)
    publishFullOctoMap(rostime);


  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_DEBUG("Map publishing in OctomapServer took %f sec", total_elapsed);

}


bool OctomapServer::octomapBinarySrv(OctomapSrv::Request  &req,
                                    OctomapSrv::Response &res)
{
  ros::WallTime startTime = ros::WallTime::now();
  ROS_INFO("Sending binary map data on service request");
  res.map.header.frame_id = m_worldFrameId;
  res.map.header.stamp = ros::Time::now();
  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  if (!octomap_msgs::binaryMapToMsg(*octree_bg, res.map))
    return false;

  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_INFO("Binary octomap sent in %f sec", total_elapsed);
  return true;
}

bool OctomapServer::octomapFullSrv(OctomapSrv::Request  &req,
                                    OctomapSrv::Response &res)
{
  ROS_INFO("Sending full map data on service request");
  res.map.header.frame_id = m_worldFrameId;
  res.map.header.stamp = ros::Time::now();

  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  if (!octomap_msgs::fullMapToMsg(*octree_bg, res.map))
    return false;

  return true;
}

bool OctomapServer::clearBBXSrv(BBXSrv::Request& req, BBXSrv::Response& resp){
  point3d min = pointMsgToOctomap(req.min);
  point3d max = pointMsgToOctomap(req.max);

  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  double thresMin = octree_bg->getClampingThresMin();
  for(OcTreeT::leaf_bbx_iterator it = octree_bg->begin_leafs_bbx(min,max),
      end=octree_bg->end_leafs_bbx(); it!= end; ++it){

    it->setLogOdds(octomap::logodds(thresMin));
    // octree_bg->updateNode(it.getKey(), -6.0f);
  }
  // TODO: eval which is faster (setLogOdds+updateInner or updateNode)
  octree_bg->updateInnerOccupancy();

  publishAll(ros::Time::now());

  return true;
}

bool OctomapServer::resetSrv(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp) {
  visualization_msgs::MarkerArray occupiedNodesVis;
  occupiedNodesVis.markers.resize(m_treeDepth +1);
  ros::Time rostime = ros::Time::now();
  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  octree_bg->clear();

  ROS_INFO("Cleared octomap");
  publishAll(rostime);

  publishBinaryOctoMap(rostime);
  for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i){

    occupiedNodesVis.markers[i].header.frame_id = m_worldFrameId;
    occupiedNodesVis.markers[i].header.stamp = rostime;
    occupiedNodesVis.markers[i].ns = "map";
    occupiedNodesVis.markers[i].id = i;
    occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
    occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
  }

  m_markerPub.publish(occupiedNodesVis);

  visualization_msgs::MarkerArray freeNodesVis;
  freeNodesVis.markers.resize(m_treeDepth +1);

  for (unsigned i= 0; i < freeNodesVis.markers.size(); ++i){

    freeNodesVis.markers[i].header.frame_id = m_worldFrameId;
    freeNodesVis.markers[i].header.stamp = rostime;
    freeNodesVis.markers[i].ns = "map";
    freeNodesVis.markers[i].id = i;
    freeNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
    freeNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
  }
  m_fmarkerPub.publish(freeNodesVis);

  return true;
}

void OctomapServer::publishBinaryOctoMap(const ros::Time& rostime) const{

  Octomap map;
  map.header.frame_id = m_worldFrameId;
  map.header.stamp = rostime;

  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  if (octomap_msgs::binaryMapToMsg(*octree_bg, map))
    m_binaryMapPub.publish(map);
  else
    ROS_ERROR("Error serializing OctoMap");
}

void OctomapServer::publishFullOctoMap(const ros::Time& rostime) const{

  Octomap map;
  map.header.frame_id = m_worldFrameId;
  map.header.stamp = rostime;

  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  if (octomap_msgs::fullMapToMsg(*octree_bg, map))
    m_fullMapPub.publish(map);
  else
    ROS_ERROR("Error serializing OctoMap");

}

bool OctomapServer::isSpeckleNode(const OcTreeKey&nKey) const {
  OcTreeKey key;
  bool neighborFound = false;
  OcTreeT* octree_bg = m_octrees.find(-1)->second;
  for (key[2] = nKey[2] - 1; !neighborFound && key[2] <= nKey[2] + 1; ++key[2]){
    for (key[1] = nKey[1] - 1; !neighborFound && key[1] <= nKey[1] + 1; ++key[1]){
      for (key[0] = nKey[0] - 1; !neighborFound && key[0] <= nKey[0] + 1; ++key[0]){
        if (key != nKey){
          OcTreeNode* node = octree_bg->search(key);
          if (node && octree_bg->isNodeOccupied(node)){
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

int main(int argc, char** argv){
  ros::init(argc, argv, "octomap_server");
  const ros::NodeHandle& private_nh = ros::NodeHandle("~");

  ros_objslampp_ycb_video::OctomapServer server;

  try{
    ros::spin();
  }catch(std::runtime_error& e){
    ROS_ERROR("octomap_server exception: %s", e.what());
    return -1;
  }

  return 0;
}

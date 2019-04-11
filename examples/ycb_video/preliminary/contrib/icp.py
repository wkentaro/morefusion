# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/Advanced/global_registration.py

import copy

import open3d
import trimesh

import objslampp


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = open3d.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    open3d.estimate_normals(
        pcd_down,
        open3d.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30),
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = open3d.compute_fpfh_feature(
        pcd_down,
        open3d.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source, target, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    result = open3d.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        distance_threshold,
        open3d.TransformationEstimationPointToPoint(False),
        4,
        [
            open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        open3d.RANSACConvergenceCriteria(4000000, 500),
    )
    return result


def refine_registration(
    source, target, source_fpfh, target_fpfh, voxel_size, transform_init
):
    distance_threshold = voxel_size * 0.4
    result = open3d.registration_icp(
        source,
        target,
        distance_threshold,
        transform_init,
        open3d.TransformationEstimationPointToPoint(False),
        open3d.ICPConvergenceCriteria(max_iteration=50),
    )
    return result


def register_cad2scan(cad, scan, debug=False):
    pcd_cad = trimesh.PointCloud(vertices=cad.vertices)
    pcd_scan = trimesh.PointCloud(vertices=scan.vertices)

    translation = (
        pcd_scan.bounding_box.centroid - pcd_cad.bounding_box.centroid
    )
    transform_init = trimesh.transformations.translation_matrix(translation)
    pcd_cad.apply_transform(transform_init)

    source = objslampp.utils.trimesh_to_open3d(pcd_cad)
    target = objslampp.utils.trimesh_to_open3d(pcd_scan)

    return register_pointcloud(source, target, debug=debug) @ transform_init


def register_pointcloud(source, target, voxel_size=0.01, debug=False):
    source = trimesh.PointCloud(vertices=source)
    target = trimesh.PointCloud(vertices=target)

    source = objslampp.utils.trimesh_to_open3d(source)
    target = objslampp.utils.trimesh_to_open3d(target)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size,
    )
    if debug:
        draw_registration_result(
            source_down,
            target_down,
            result.transformation,
        )

    result = refine_registration(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        voxel_size,
        result.transformation,
    )
    if debug:
        draw_registration_result(
            source_down,
            target_down,
            result.transformation,
        )

    return result.transformation

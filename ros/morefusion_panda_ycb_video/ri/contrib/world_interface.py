# flake8: noqa
import numpy as _np
import rospy as _rospy
from morefusion_panda_ycb_video.srv import UpdateScene as _UpdateScene
from geometry_msgs.msg import Point as _Point
from shape_msgs.msg import Mesh as _Mesh
from shape_msgs.msg import MeshTriangle as _MeshTriangle
from moveit_msgs.msg import CollisionObject as _CollisionObject
from moveit_msgs.msg import AttachedCollisionObject as _AttachedCollisionObject

# ToDo get this working with touch links
def add_static_meshes(mesh_ids, meshes, poses):

    collision_objects = list()

    for mesh_id, mesh, pose in zip(mesh_ids, meshes, poses):

        collision_object_msg = _CollisionObject()

        triangles = list()
        for face in _np.array(mesh.faces):
            triangle = _MeshTriangle()
            triangle.vertex_indices = face
            triangles.append(triangle)

        vertices = list()
        for vertex in _np.array(mesh.vertices):
            point = _Point()
            point.x = vertex[0]
            point.y = vertex[1]
            point.z = vertex[2]
            vertices.append(point)

        mesh_msg = _Mesh()
        mesh_msg.vertices = vertices
        mesh_msg.triangles = triangles

        collision_object_msg.meshes.append(mesh_msg)
        collision_object_msg.mesh_poses.append(pose)
        collision_object_msg.id = mesh_id
        collision_object_msg.operation = collision_object_msg.ADD

        collision_objects.append(collision_object_msg)

    _rospy.wait_for_service("update_scene")
    update_scene = _rospy.ServiceProxy("update_scene", _UpdateScene)
    response = update_scene(list(), collision_objects)
    return response.success


def remove_static_meshes(mesh_ids):

    collision_objects = list()

    for mesh_id in mesh_ids:

        collision_object_msg = _CollisionObject()
        collision_object_msg.id = mesh_id
        collision_object_msg.operation = collision_object_msg.REMOVE
        collision_objects.append(collision_object_msg)

    _rospy.wait_for_service("update_scene")
    update_scene = _rospy.ServiceProxy("update_scene", _UpdateScene)
    response = update_scene(list(), collision_objects)
    return response.success


def add_attached_meshes(mesh_ids, meshes, poses, link_names):

    attached_objects = list()

    for mesh_id, mesh, pose, link_name in zip(
        mesh_ids, meshes, poses, link_names
    ):

        attached_object_msg = _AttachedCollisionObject()
        attached_object_msg.link_name = link_name
        attached_object_msg.touch_links = [
            "panda_suction_cup",
            "panda_table",
            "panda_target_box",
        ]

        triangles = list()
        for face in _np.array(mesh.faces):
            triangle = _MeshTriangle()
            triangle.vertex_indices = face
            triangles.append(triangle)

        vertices = list()
        for vertex in _np.array(mesh.vertices):
            point = _Point()
            point.x = vertex[0]
            point.y = vertex[1]
            point.z = vertex[2]
            vertices.append(point)

        mesh_msg = _Mesh()
        mesh_msg.vertices = vertices
        mesh_msg.triangles = triangles

        attached_object_msg.object.meshes.append(mesh_msg)

        attached_object_msg.object.mesh_poses.append(pose)
        attached_object_msg.object.id = mesh_id
        attached_object_msg.object.operation = attached_object_msg.object.ADD

        attached_objects.append(attached_object_msg)

    _rospy.wait_for_service("update_scene")
    update_scene = _rospy.ServiceProxy("update_scene", _UpdateScene)
    response = update_scene(attached_objects, list())
    return response.success


def remove_attached_meshes(mesh_ids, link_names):

    attached_objects = list()

    for mesh_id, link_name in zip(mesh_ids, link_names):

        attached_object_msg = _AttachedCollisionObject()
        attached_object_msg.object.id = mesh_id
        attached_object_msg.link_name = link_name
        attached_object_msg.object.operation = (
            attached_object_msg.object.REMOVE
        )

        attached_objects.append(attached_object_msg)

    _rospy.wait_for_service("update_scene")
    update_scene = _rospy.ServiceProxy("update_scene", _UpdateScene)
    response = update_scene(attached_objects, list())
    return response.success

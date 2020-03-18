#!/usr/bin/env python

import morefusion

import contrib

import check_iterative_closest_point_link
import check_iterative_collision_check_link
from visualize_data import visualize_data


def get_scenes():
    scenes = visualize_data()
    # scenes.pop("grid_target")
    scenes.pop("grid_nontarget_empty")
    scenes.pop("cad")

    scenes_icp = check_iterative_closest_point_link.get_scenes()
    scenes_icc = check_iterative_collision_check_link.get_scenes()
    for i, scene_icc in enumerate(scenes_icc):
        scene_icp = next(scenes_icp)
        scenes["icp"] = scene_icp["icp"]
        scenes["icc"] = scene_icc["icc"]
        if i == 0:
            yield from contrib.move_camera_auto(scenes, motion_id=0)
        yield scenes
    yield from contrib.move_camera_auto(scenes, motion_id=1)


if __name__ == "__main__":
    scenes = get_scenes()
    morefusion.extra.trimesh.display_scenes(scenes, tile=(2, 2))

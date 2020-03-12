import functools
import re

import imgviz
import numpy as np
import path
import trimesh
import trimesh.transformations as ttf
import yaml

import morefusion


here = path.Path(__file__).abspath().parent


def grid(scale=1):
    path = []
    N = 10
    for i in range(2):
        for x in range(-N, N + 1):
            if i == 0:
                path.append(trimesh.load_path([[x, -N], [x, N]]))
            else:
                path.append(trimesh.load_path([[-N, x], [N, x]]))
    path = functools.reduce(lambda x, y: x + y, path)
    path.apply_scale(scale)
    return path


def get_data():
    data = {}

    data["instances"] = []
    for npz_file in sorted((here / "data").listdir()):
        if not re.match(r"[0-9]+.npz", npz_file.basename()):
            continue
        instance = dict(np.load(npz_file))
        instance["id"] = int(npz_file.basename().stem)
        data["instances"].append(instance)

    data["rgb"] = imgviz.io.imread("data/image.png")

    depth = np.load("data/depth.npz")["arr_0"]
    depth = depth.astype(np.float32) / 1000
    data["depth"] = depth

    with open(f"data/camera_info.yaml") as f:
        camera_info = yaml.safe_load(f)
    data["intrinsic_matrix"] = np.array(camera_info["K"]).reshape(3, 3)

    data["T_cinematic2world"] = np.array(
        [
            [0.65291082, -0.10677561, 0.74987094, -0.42925002],
            [0.75528109, 0.166384, -0.63392968, 0.3910296],
            [-0.0570783, 0.98026289, 0.18927951, 0.48834561],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return data


def move_camera_auto(scenes, motion_id):

    transforms = np.array(
        [
            [
                [0.65291082, -0.10677561, 0.74987094, -0.42925002],
                [0.75528109, 0.166384, -0.63392968, 0.3910296],
                [-0.0570783, 0.98026289, 0.18927951, 0.48834561],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.99762283, 0.04747429, -0.04994869, 0.04963055],
                [-0.04687166, -0.06385564, -0.99685781, 0.5102746],
                [-0.05051463, 0.9968293, -0.06147865, 0.63364497],
                [0.0, 0.0, 0.0, 1.0],
            ],
            np.eye(4),
        ]
    )
    if motion_id == 0:
        transforms = transforms[[0]]
    elif motion_id == 1:
        transforms = transforms[[1, 2, 0]]
    else:
        raise ValueError

    transform_prev = morefusion.extra.trimesh.from_opengl_transform(
        list(scenes.values())[0].camera_transform
    )
    for transform_next in transforms:
        point_prev = np.r_[
            ttf.translation_from_matrix(transform_prev),
            ttf.euler_from_matrix(transform_prev),
        ]
        point_next = np.r_[
            ttf.translation_from_matrix(transform_next),
            ttf.euler_from_matrix(transform_next),
        ]

        for w in np.linspace(0, 1, 50):
            point = point_prev + w * (point_next - point_prev)
            transform = ttf.translation_matrix(point[:3]) @ ttf.euler_matrix(
                *point[3:]
            )
            for scene in scenes.values():
                scene.camera_transform[
                    ...
                ] = morefusion.extra.trimesh.to_opengl_transform(transform)
            yield scenes

        transform_prev = transform_next

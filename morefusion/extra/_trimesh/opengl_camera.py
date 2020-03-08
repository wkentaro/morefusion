import math
import trimesh


class OpenGLCamera(trimesh.scene.Camera):

    """Camera object given the camera resolution and fovy.

    Parameters
    ----------
    resolution: tuple
        Image size: (width, height).
    fovy: float
        Field of view of y-axis in degrees.
    """

    def __init__(self, resolution, fovy):
        width, height = resolution
        aspect_ratio = 1.0 * width / height
        fovy = math.radians(fovy)
        fovx = 2 * math.atan(math.tan(fovy * 0.5) * aspect_ratio)
        fov = (math.degrees(fovx), math.degrees(fovy))
        super().__init__(resolution=resolution, fov=fov)

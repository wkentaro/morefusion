import io

import PIL.Image
import pyglet


def numpy_to_image(arr):
    with io.BytesIO() as f:
        PIL.Image.fromarray(arr).save(f, format="PNG")
        return pyglet.image.load(filename=None, file=f)

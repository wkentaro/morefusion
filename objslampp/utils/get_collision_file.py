import pathlib
import shlex
import subprocess


def get_collision_file(visual_file):
    visual_file = pathlib.Path(visual_file)
    name = visual_file.name
    name_noext, ext = name.rsplit('.')
    collision_file = visual_file.parent / (name_noext + '_convex.' + ext)
    if not collision_file.exists():
        cmd = f'testVHACD --input {visual_file} --output {collision_file}'\
              ' --log /tmp/testVHACD.log --resolution 200000'
        # print(f'+ {cmd}')
        subprocess.check_call(shlex.split(cmd))
    return collision_file

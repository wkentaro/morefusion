import shlex
import subprocess

import path


def get_collision_file(visual_file):
    visual_file = path.Path(visual_file)
    collision_file = visual_file.stripext() + '_convex' + visual_file.ext
    if not collision_file.exists():
        cmd = f'testVHACD --input {visual_file} --output {collision_file}'\
              ' --log /tmp/testVHACD.log --resolution 200000'
        # print(f'+ {cmd}')
        subprocess.check_output(shlex.split(cmd))
    return collision_file

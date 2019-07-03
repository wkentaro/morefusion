import shlex
import subprocess

import path


def get_binvox_file(cad_file, solid=True):
    cad_file = path.Path(cad_file)
    vox_file = cad_file.with_suffix('.binvox')
    if vox_file.exists():
        raise IOError(f'Binvox file exists: {vox_file}')

    out_file = cad_file.with_suffix('.solid.binvox')
    if not out_file.exists():
        cmd = f'binvox -d 128 -aw -dc -down -pb {cad_file}'
        subprocess.check_output(shlex.split(cmd))
        vox_file.rename(out_file)
    return out_file

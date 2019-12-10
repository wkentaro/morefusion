import os
import subprocess


def githash(filename=None):
    if filename is None:
        cwd = None
    else:
        cwd = os.path.dirname(os.path.abspath(filename))
    cmd = 'git log -1 --format="%h"'
    try:
        return subprocess.check_output(
            cmd, shell=True, cwd=cwd
        ).decode().strip()
    except Exception:
        return

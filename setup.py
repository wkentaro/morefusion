from setuptools import find_packages
from setuptools import setup
import shlex
import subprocess


def git_version():
    cmd = 'git log --format="%h" -n 1'
    return subprocess.check_output(shlex.split(cmd)).decode().strip()


version = git_version()

install_requires = []
with open('requirements.txt') as f:
    for req in f:
        if req.startswith('-e'):
            continue
        install_requires.append(req.strip())

setup(
    name='objslampp',
    version=version,
    packages=find_packages(),
    install_requires=install_requires,
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    license='MIT',
    url='https://github.com/wkentaro/objslampp',
    description='Volumetric fusion and CAD alignment for object-level SLAM',
)

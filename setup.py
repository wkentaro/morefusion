import re
from setuptools import find_packages
from setuptools import setup


def get_version():
    filename = "morefusion/__init__.py"
    with open(filename) as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError(f"{filename} doesn't contain __version__")
    version = match.groups()[0]
    return version


def get_install_requires():
    install_requires = []
    with open("requirements.txt") as f:
        for req in f:
            install_requires.append(req.strip())
    return install_requires


setup(
    name="morefusion",
    version=get_version(),
    packages=find_packages(),
    install_requires=get_install_requires(),
    author="Kentaro Wada",
    author_email="www.kentaro.wada@gmail.com",
    license="MIT",
    url="https://github.com/wkentaro/morefusion",
    description="Multi-object reasoning for 6d pose estimation from volumetric fusion",  # NOQA
)

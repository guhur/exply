from setuptools import setup, find_packages
import os


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="exply",
    version="0.1",
    url="https://github.com/guhur/exply",
    long_description=readme(),
    license="GNU GPLv3",
    author="Pierre-Louis Guhur",
    author_email="pierre-louis.guhur@inria.fr",
    description="Management of Tensorboard and experiments with PyTorch",
    packages=find_packages(exclude=["tests", "doc"]),
    zip_safe=False,
    include_package_data=True,
    install_requires=[],
)

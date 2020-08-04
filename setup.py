#!/usr/bin/env python
import os
from setuptools import setup
from setuptools import find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext

with open("README.md", "r") as f:
    readme = f.read()

def _requires_from_file(filename):
    return open(filename).read().splitlines()

ext_modules = [
    Extension('nsmr.envs.obs.raycasting', sources=['nsmr/envs/obs/raycasting.pyx'])
]

setup(
    name='nsmr',
    version='0.2.0',
    url='https://github.com/Jumpei-Arima/Navigation_Simulator_for_Mobile_Robot',
    author='Jumpei Arima',
    author_email='arijun0307@gmail.com',
    maintainer='Jumpei Arima',
    maintainer_email='arijun0307@gmail.com',
    description='Navigation Simulator for Mobile Robot',
    long_description=readme,
    python_requires='>3.5.0',
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=_requires_from_file('requirements.txt'),
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)

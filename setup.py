#!/usr/bin/env python
"""
# Author: Lai Wenpu
# Created Time : Sun 28 Apr 2024
"""
import pathlib
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name='mist',
      version=get_version(HERE / "mist/__init__.py"),
      packages=find_packages(),
      description='',
      long_description=README,

      author='Lai Wenpu',
      author_email='kyzy850520@163.com',
      url='https://github.com/aapupu/MIST',
      scripts=['MIST.py'],
      install_requires=requirements,
      python_requires='>3.7',
      license='MIT',

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7.3',
          # 'Operating System :: MacOS :: MacOS X',
          # 'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Framework :: PyTorch',
          'Environment :: Console',
     ],
     )

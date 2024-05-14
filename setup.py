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

setup(
        name='mist',
        version='1.0.0',
        author='Lai Wenpu',
        author_email='kyzy850520@163.com',
        description='MIST: Deep Learning Integration of Single-Cell RNA and TCR Sequencing Data for T-Cell Insight',
        long_description=README,
        packages=find_packages(),
        install_requires=[requirements], # 
        package_data={
        'MIST': ['doc/*.csv'], 
        },
        url='https://github.com/aapupu/MIST',
        scripts=['MIST.py'],
        python_requires='>=3.7.3',
        license='GPL-3.0',
        
        classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GPL-3.0 License',
          'Programming Language :: Python :: 3.7.3',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Framework :: PyTorch',
          'Environment :: Console',
        ]
)

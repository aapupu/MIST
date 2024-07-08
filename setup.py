from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='mist-tcr',
    version='1.0.1',
    author='Lai Wenpu',
    author_email='kyzy850520@163.com',
    description='MIST: an interpretable and flexible deep learning framework for single-T cell transcriptome and receptor analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    license_files=('LICENSE',),
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "numpy>=1.21.6",
        "pandas>=1.3.5",
        "anndata>=0.8.0",
        "scipy>=1.7.3",
        "scikit-learn>=1.0.2",
        "scanpy>=1.9.3",
        "scirpy>=0.12.0",
        "torch>=1.13.0",
        "einops>=0.6.0",
        "tqdm>=4.64.1",
        "prefetch_generator>=1.0.3",
        "local-attention>=1.5.7",
        "leidenalg>=0.10.2"
    ],
    packages=find_packages(include=['mist', 'mist.*']),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'MIST=mist.main:main',
        ],
    },
    project_urls={
        'Homepage': 'https://github.com/aapupu/MIST',
    },
    package_data={
        '': ['doc/**'],
    },
)

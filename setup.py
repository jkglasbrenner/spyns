#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import setuptools
from sphinx.setup_command import BuildDoc

with open('README.rst', 'rt', encoding='utf8') as f:
    readme = f.read()

name = "isingmodel"
version = "0.1"
release = "0.1.0"

setuptools.setup(
    name=name,
    author="James K. Glasbrenner",
    author_email="jglasbr2@gmu.edu",
    license="MIT",
    version=release,
    url="https://github.com/jkglasbrenner/ising-model-py",
    project_urls=OrderedDict((
        ("Documentation", "https://isingmodel.readthedocs.io"),
        ("Code", "https://github.com/jkglasbrenner/ising-model-py"),
    )),
    description="The two-dimensional Ising model implemented in Python.",
    long_description=readme,
    python_requires=">=3.7",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
        ],
    },
    cmdclass={"build_sphinx": BuildDoc},
    command_options={
        "build_sphinx": {
            "project": ("setup.py", name),
            "version": ("setup.py", version),
            "release": ("setup.py", release),
            "source_dir": ("setup.py", "docs"),
            "build_dir": ("setup.py", "docs/_build"),
        }
    },
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import sys

import setuptools
from sphinx.setup_command import BuildDoc

with open('README.rst', 'rt', encoding='utf8') as f:
    readme = f.read()

name = "spyns"
version = "0.1"
release = "0.1.0"

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

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
    description="Monte Carlo simulations of magnetic systems in Python.",
    long_description=readme,
    python_requires=">=3.7",
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=[] + pytest_runner,
    install_requires=[
        "numpy",
        "pandas",
        "pymatgen",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
        ],
    },
    tests_require=[
        "pytest",
    ],
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

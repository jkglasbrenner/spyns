#!/usr/bin/env python

import os
import sys
from collections import OrderedDict

import setuptools

USE_CYTHON = os.environ.get("USE_CYTHON", None)


with open("README.rst", "rt", encoding="utf8") as f:
    readme = f.read()

name = "spyns"
version = "0.1"
release = "0.1.0"

dependencies = ["numpy>=1.16.2", "pandas>=0.24.2", "pymatgen>=2019.5.8"]
cmdclass = {}
extras_dependencies = {
    "docs": [
        "sphinx>=1.8.5",
        "sphinx-rtd-theme==0.4.3",
        "sphinx-autodoc-typehints==1.6.0",
    ],
    "dev": [
        "autopep8==1.4.4",
        "black==19.3b0",
        "Cython>=0.29.6",
        "entrypoints==0.3",
        "flake8-bugbear==19.3.0",
        "flake8==3.7.7",
        "ipython==7.4.0",
        "mypy==0.710",
        "pre-commit==1.17.0",
        "pydocstyle==3.0.0",
        "pytoml==0.1.20",
        "seed-isort-config==1.9.1",
    ],
}
tests_dependencies = (["pytest==4.3.1", "pytest-runner==4.4"],)

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

setup_requires = [] + pytest_runner

ext = ".pyx" if USE_CYTHON else ".cpp"

extensions = [
    setuptools.extension.Extension(
        "spyns.algorithms.metropolis.base_cython",
        sources=["spyns/algorithms/metropolis/base_cython" + ext],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    setuptools.extension.Extension(
        "spyns.algorithms.metropolis.heisenberg_cython",
        sources=["spyns/algorithms/metropolis/heisenberg_cython" + ext],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    setuptools.extension.Extension(
        "spyns.random_numbers.distribution",
        sources=["spyns/random_numbers/distribution" + ext],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    setuptools.extension.Extension(
        "spyns.model.heisenberg_cython",
        sources=["spyns/model/heisenberg_cython" + ext],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    setuptools.extension.Extension(
        "spyns.data_cython",
        sources=["spyns/data_cython" + ext],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions, annotate=True)

try:
    from sphinx.setup_command import BuildDoc

    cmdclass["build_sphinx"] = BuildDoc

except ImportError:
    print("WARNING: sphinx not available, not building docs")

setuptools.setup(
    name=name,
    author="James K. Glasbrenner",
    author_email="jglasbr2@gmu.edu",
    license="MIT",
    version=release,
    url="https://github.com/jkglasbrenner/spyns",
    project_urls=OrderedDict(
        (
            ("Documentation", "https://spyns.readthedocs.io"),
            ("Code", "https://github.com/jkglasbrenner/spyns"),
        )
    ),
    description="Monte Carlo simulations of magnetic systems in Python.",
    long_description=readme,
    python_requires=">=3.7",
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=setup_requires,
    ext_modules=extensions,
    zip_safe=False,
    install_requires=dependencies,
    extras_require=extras_dependencies,
    tests_require=tests_dependencies,
    cmdclass=cmdclass,
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

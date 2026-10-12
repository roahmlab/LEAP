#!/usr/bin/env python3
"""Build the public StageFunction adapter for generated C residual models.

    .venv_comp/bin/python aligator_bench/setup_sparse.py build_ext --inplace

Produces aligator_bench/_generated_sparse*.so. Every evaluation buffer is
owned by its stage Data object so the residuals run without the Python GIL.
This module contains no solver implementation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFIX = Path(sys.prefix)
LIBDIR = PREFIX / "lib"
PY_TAG = f"{sys.version_info.major}{sys.version_info.minor}"

boost_python = sorted(LIBDIR.glob(f"libboost_python{PY_TAG}.so*"))
if not boost_python:
    raise RuntimeError(f"no Boost.Python {PY_TAG} library found under {LIBDIR}")
boost_python_name = boost_python[0].name.split(".so", 1)[0].removeprefix("lib")

# build_ext --inplace resolves the package destination from the current dir.
os.chdir(ROOT)

extension = Extension(
    "aligator_bench._generated_sparse",
    sources=[str(HERE / "generated_sparse.cpp")],
    include_dirs=[
        str(PREFIX / "include"),
        # pinocchio 4.1 reorganized its headers; this shim tree is what
        # aligator's own <pinocchio/algorithm/...> includes resolve against.
        str(PREFIX / "include/pinocchio/deprecated"),
        str(PREFIX / "include/eigen3"),
        np.get_include(),
    ],
    library_dirs=[str(LIBDIR)],
    runtime_library_dirs=[str(LIBDIR)],
    libraries=["aligator", "pinocchio_default", "pinocchio_parsers",
               "eigenpy", "dl", boost_python_name],
    # exactly pinocchio's exported INTERFACE_COMPILE_DEFINITIONS (see
    # lib/cmake/pinocchio/pinocchioTargets.cmake) -- its 25-type
    # JointCollection variant overflows boost::mpl's default limits
    # otherwise -- plus the two aligator flags.
    define_macros=[("BOOST_MPL_LIMIT_LIST_SIZE", "30"),
                   ("BOOST_MPL_LIMIT_VECTOR_SIZE", "30"),
                   ("BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS", None),
                   ("BOOST_FUSION_INVOKE_MAX_ARITY", "12"),
                   ("ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION", "1"),
                   ("ALIGATOR_PINOCCHIO_V4", "1")],
    extra_compile_args=["-O2", "-std=c++17", "-fopenmp",
                        "-Wno-deprecated-declarations"],
    extra_link_args=["-fopenmp"],
    language="c++",
)

setup(
    name="aligator-bench-generated-residual",
    version="0.1.0",
    ext_modules=[extension],
    script_args=sys.argv[1:],
)

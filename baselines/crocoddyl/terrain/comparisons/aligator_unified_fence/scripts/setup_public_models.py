#!/usr/bin/env python3
"""Build physical residuals supplied to ALIGATOR's public model interface."""
import os
from pathlib import Path
import sys
import numpy as np
from setuptools import setup,Extension
here=Path(__file__).resolve().parent;prefix=Path(sys.prefix)
os.chdir(here)
tag=f'{sys.version_info.major}{sys.version_info.minor}'
setup(name='aligator-fence-public-models',version='0.1',ext_modules=[Extension(
 '_public_models',sources=[str(here/'public_models.cpp')],
 include_dirs=[str(prefix/'include'),str(prefix/'include/pinocchio/deprecated'),str(prefix/'include/eigen3'),np.get_include()],
 library_dirs=[str(prefix/'lib')],runtime_library_dirs=[str(prefix/'lib')],
 libraries=['aligator','pinocchio_default','pinocchio_collision','coal','eigenpy',f'boost_python{tag}'],
 define_macros=[('BOOST_MPL_LIMIT_LIST_SIZE','30'),('BOOST_MPL_LIMIT_VECTOR_SIZE','30'),
 ('BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS',None),('BOOST_FUSION_INVOKE_MAX_ARITY','12'),
 ('PINOCCHIO_WITH_COLLISION','1'),('PINOCCHIO_WITH_HPP_FCL','1'),
 ('ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION','1'),('ALIGATOR_PINOCCHIO_V4','1')],
 extra_compile_args=['-O2','-std=c++17','-Wno-deprecated-declarations'],language='c++')])

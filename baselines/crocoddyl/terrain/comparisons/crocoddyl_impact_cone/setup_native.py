"""Build only public modeling/callback adapters; never rebuild or patch a solver."""
from pathlib import Path
import os,sys,numpy as np
from setuptools import setup,Extension
here=Path(__file__).resolve().parent;prefix=Path(sys.prefix);os.chdir(here)
setup(name='crocoddyl-impact-cone-models',version='1.0',ext_modules=[Extension('_impact_native',
 sources=[str(here/'native.cpp')],include_dirs=[str(prefix/'include'),str(prefix/'include/pinocchio/deprecated'),str(prefix/'include/eigen3'),np.get_include()],
 library_dirs=[str(prefix/'lib')],runtime_library_dirs=[str(prefix/'lib')],
 libraries=['crocoddyl','aligator','pinocchio_default','pinocchio_collision','coal','eigenpy',f'boost_python{sys.version_info.major}{sys.version_info.minor}'],
 define_macros=[('BOOST_MPL_LIMIT_LIST_SIZE','30'),('BOOST_MPL_LIMIT_VECTOR_SIZE','30'),('BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS',None),('BOOST_FUSION_INVOKE_MAX_ARITY','12'),('PINOCCHIO_WITH_COLLISION','1'),('PINOCCHIO_WITH_HPP_FCL','1'),('ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION','1'),('ALIGATOR_PINOCCHIO_V4','1')],
 extra_compile_args=['-O2','-std=c++17','-fopenmp','-Wno-deprecated-declarations'],extra_link_args=['-fopenmp'],language='c++')])

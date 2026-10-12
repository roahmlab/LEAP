cmake_minimum_required(VERSION 3.22)
project(leap2 LANGUAGES CXX)

# LEAP: gradient-flow augmented-Lagrangian trajectory optimization for contact
# constrained robots. One static library `leap` (modules: spectral, model, problem,
# alm, linalg, solver, tasks, io), one driver `leap_solve`, and the tests.

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(LEAP_WERROR "Treat warnings as errors" OFF)
option(LEAP_IPO "Interprocedural optimization in Release builds" ON)
option(LEAP_OPENMP "Parallel per-node assembly via OpenMP" ON)
option(LEAP_CHOLMOD "SuiteSparse CHOLMOD sparse Cholesky for the separator system" ON)
option(LEAP_LAPACK "LAPACK dense Cholesky for the element blocks" ON)

find_package(Eigen3 REQUIRED CONFIG)
find_package(pinocchio REQUIRED CONFIG)
find_path(CPPAD_INCLUDE_DIR cppad/cppad.hpp REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
if(LEAP_OPENMP)
  find_package(OpenMP)
endif()

if(LEAP_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT LEAP_IPO_SUPPORTED OUTPUT LEAP_IPO_ERROR LANGUAGES CXX)
  if(LEAP_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  endif()
endif()

# ---------------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------------
set(LEAP_MODULES spectral model problem alm linalg solver tasks io)
set(LEAP_SOURCES)
foreach(m ${LEAP_MODULES})
  file(GLOB m_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${m}/src/*.cpp)
  list(APPEND LEAP_SOURCES ${m_sources})
endforeach()

add_library(leap STATIC ${LEAP_SOURCES})
foreach(m ${LEAP_MODULES})
  target_include_directories(leap PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${m}/include)
endforeach()
target_include_directories(leap SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(leap SYSTEM PUBLIC ${CPPAD_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
target_link_libraries(leap PUBLIC Eigen3::Eigen pinocchio::pinocchio ${HDF5_LIBRARIES})
target_compile_options(leap PRIVATE -Wall -Wextra)
if(LEAP_WERROR)
  target_compile_options(leap PRIVATE -Werror)
endif()
if(LEAP_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(leap PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(leap PUBLIC LEAP_HAS_OPENMP)
  message(STATUS "leap: OpenMP enabled")
endif()

if(LEAP_CHOLMOD)
  find_path(LEAP_CHOLMOD_INCLUDE cholmod.h PATH_SUFFIXES suitesparse)
  find_library(LEAP_CHOLMOD_LIB cholmod)
  if(LEAP_CHOLMOD_INCLUDE AND LEAP_CHOLMOD_LIB)
    message(STATUS "leap: CHOLMOD found (${LEAP_CHOLMOD_LIB})")
    target_include_directories(leap SYSTEM PUBLIC ${LEAP_CHOLMOD_INCLUDE})
    target_link_libraries(leap PUBLIC ${LEAP_CHOLMOD_LIB})
    target_compile_definitions(leap PUBLIC LEAP_HAS_CHOLMOD)
  else()
    message(STATUS "leap: CHOLMOD not found - the Eigen sparse Cholesky is used instead")
  endif()
endif()
if(LEAP_LAPACK)
  if(LEAP_CHOLMOD_LIB)
    get_filename_component(LEAP_CHOLMOD_LIBDIR ${LEAP_CHOLMOD_LIB} DIRECTORY)
  endif()
  find_library(LEAP_LAPACK_LIB lapack HINTS ${LEAP_CHOLMOD_LIBDIR})
  find_library(LEAP_BLAS_LIB blas HINTS ${LEAP_CHOLMOD_LIBDIR})
  if(LEAP_LAPACK_LIB AND LEAP_BLAS_LIB)
    message(STATUS "leap: LAPACK found (${LEAP_LAPACK_LIB})")
    target_link_libraries(leap PUBLIC ${LEAP_LAPACK_LIB} ${LEAP_BLAS_LIB})
    target_compile_definitions(leap PUBLIC LEAP_HAS_LAPACK)
  else()
    message(STATUS "leap: LAPACK not found - Eigen dense Cholesky is used instead")
  endif()
endif()

# ---------------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------------
# examples/: one executable per source, written against the formulation API
file(GLOB LEAP_EXAMPLE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
foreach(src ${LEAP_EXAMPLE_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE leap)
  target_include_directories(${name} SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
endforeach()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps/leap_solve.cpp)
  add_executable(leap_solve apps/leap_solve.cpp)
  target_link_libraries(leap_solve PRIVATE leap)
  target_include_directories(leap_solve SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
  target_compile_options(leap_solve PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------------
# tests (standalone executables; `ctest` runs them)
# ---------------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# Every test is one executable. `add_leap_test(name [args...])` builds tests/<name>.cpp,
# links the library and registers it with ctest. Tests that need the repository
# (URDFs, robot yamls) receive the repository root as their first argument.

set(LEAP_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

function(add_leap_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE leap)
  target_include_directories(${name} SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction()

file(GLOB LEAP_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
foreach(src ${LEAP_TEST_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  add_leap_test(${name} ${LEAP_REPO_ROOT})
endforeach()

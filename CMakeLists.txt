cmake_minimum_required(VERSION 2.8)
project(complete3D)

set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(Eigen3 REQUIRED)

include_directories(
    src/dfgen
    src/utils
    src/vox2mesh
    ${EIGEN3_INCLUDE_DIR})

add_executable(dfgen src/dfgen/main.cpp src/dfgen/makelevelset3.cpp)
add_executable(vox2mesh src/vox2mesh/main.cpp)
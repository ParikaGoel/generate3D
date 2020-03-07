cmake_minimum_required(VERSION 2.8)
project(complete3D)

set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(Eigen3 REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)

include_directories(
    src/dfgen
    src/utils
    src/vox2mesh
    ${EIGEN3_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS})

add_executable(dfgen src/dfgen/main.cpp src/dfgen/makelevelset3.cpp)
target_link_libraries(dfgen ${HDF5_CXX_LIBRARIES})

add_executable(vox2mesh src/vox2mesh/main.cpp)
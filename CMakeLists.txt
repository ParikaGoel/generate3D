cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME complete3D)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(Eigen3 3.3 REQUIRED)

# Adding include directories
include_directories(
        src/utils
        src/dfgen
        src/vox2mesh
        ${EIGEN3_INCLUDE_DIR}
        )

set(SOURCE_FILES
        src/utils/tiny_obj_loader.cpp
        src/utils/tinyply.cpp
        src/utils/camera.cpp
        src/dfgen/dfgen.cpp
        src/dfgen/makelevelset3.cpp
        src/vox2mesh/vox2mesh.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} src/main.cpp)

target_link_libraries(${PROJECT_NAME}
        ${EIGEN3_LIBS}
        )
cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME complete3D)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(Eigen3 3.3 REQUIRED)
find_package(Open3D)

if(Open3D_FOUND)
        message(STATUS "Found Open3D ${Open3D_VERSION}")

        list(APPEND Open3D_LIBRARIES dl)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

        if(UNIX)
                add_definitions(-DUNIX)
                add_compile_options(-Wno-deprecated-declarations)
                add_compile_options(-Wno-unused-result)
                add_definitions(-O3)
        endif()

        # link_directories must be before add_executable
        link_directories(${Open3D_LIBRARY_DIRS})
endif()

# Adding include directories
include_directories(
        src/utils
        src/dfgen
        src/vox2mesh
        ${EIGEN3_INCLUDE_DIR}
        )

set(SOURCE_FILES
        src/dfgen/dfgen.cpp
        src/dfgen/makelevelset3.cpp
        src/vox2mesh/vox2mesh.cpp
        src/utils/tiny_obj_loader.cpp
        src/utils/tinyply.cpp
        src/utils/image.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} src/main.cpp)

target_link_libraries(${PROJECT_NAME}
        ${EIGEN3_LIBS}
        )

if(Open3D_FOUND)
        target_link_libraries(${PROJECT_NAME} ${Open3D_LIBRARIES})

        target_include_directories(${PROJECT_NAME} PUBLIC ${Open3D_INCLUDE_DIRS})
endif()
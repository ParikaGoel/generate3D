cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME complete3D)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)

# Adding include directories
include_directories(src/utils)

# Add source files
#set(${PROJECT_NAME}_SOURCES
#        src/mLibInclude.cpp)

add_executable(${PROJECT_NAME} src/main.cpp)
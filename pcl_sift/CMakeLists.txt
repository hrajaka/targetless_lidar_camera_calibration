cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)

project(sift)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(sift sift.cpp)

target_link_libraries(sift ${PCL_LIBRARIES})


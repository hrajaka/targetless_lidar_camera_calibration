cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(lidar_segmentation)

find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (lidar_seg lidar_seg.cpp)
target_link_libraries (lidar_seg ${PCL_LIBRARIES})

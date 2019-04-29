cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)

project(pointcloud)

find_package(OpenCV REQUIRED)
find_package(PCL 1.2 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(projection projection.cpp)

target_link_libraries(projection ${OpenCV_LIBS} ${PCL_LIBRARIES})


cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(feature_tracking)

find_package(OpenCV 4.5 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable(feature_tracker src/featurePipeline.cpp)
target_link_libraries (feature_tracker ${OpenCV_LIBRARIES})

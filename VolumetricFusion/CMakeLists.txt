cmake_minimum_required(VERSION 3.12.2)

project(VolumetricFusion)

if (APPLE)
  message(STATUS "Compiling for Mac OS")
  include_directories(
    /usr/local/Cellar/librealsense/2.29.0/include 
    /usr/local/Cellar/glfw/3.3/include
  )
  add_definitions(-DAPPLE)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif (APPLE)

add_subdirectory(align)
add_subdirectory(align-advanced)
add_subdirectory(ar-basic)
add_subdirectory(callback)
add_subdirectory(capture)
add_subdirectory(hello-realsense)
add_subdirectory(measure)
add_subdirectory(motion)
add_subdirectory(multicam)
add_subdirectory(pointcloud)
add_subdirectory(pose)
add_subdirectory(pose-predict)
add_subdirectory(post-processing)
add_subdirectory(record-playback)
add_subdirectory(save-to-disk)
add_subdirectory(sensor-control)
add_subdirectory(software-device)
add_subdirectory(trajectory)
add_subdirectory(VolumetricFusion)

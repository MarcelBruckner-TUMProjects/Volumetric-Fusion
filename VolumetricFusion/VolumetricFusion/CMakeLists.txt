cmake_minimum_required(VERSION 3.12.2)

project(VolumetricFusion)

if (APPLE)
  message(STATUS "Compiling for Mac OS")
  find_package(realsense2 REQUIRED)
  find_package(glfw3 REQUIRED)
  find_package(OpenGL REQUIRED)
  include_directories(
    /usr/local/Cellar/librealsense/2.29.0/include 
    /usr/local/Cellar/glfw/3.3/include
  )
  #set_source_files_properties(
   #MainWindow.cpp
   #Recorder.cpp
   #CaptureDevice.cpp)
#  set(SOURCE_FILES
#    VolumetricFusion/MainWindow.cpp
#    VolumetricFusion/Recorder.cpp
#    VolumetricFusion/CaptureDevice.cpp)
#  add_executable(VolumetricFusion ${SOURCE_FILES})
  add_definitions(-DAPPLE)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif (APPLE)

add_executable(VolumetricFusion
  MainWindow.cpp
  Recorder.cpp
  CaptureDevice.cpp
)

target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY})
target_link_libraries(${PROJECT_NAME} glfw ${GLFW_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OPENGL_gl_LIBRARY})



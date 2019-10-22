cmake_minimum_required(VERSION 3.12.2)

project(VolumetricFusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#add_compile_options(-Wall -Wextra)
set(CMAKE_C_FLAGS_DEBUG "-D_DEBUG")
set(CMAKE_BUILD_TYPE Debug)

if (APPLE)
  message(STATUS "Compiling for Mac OS")
  find_package(realsense2 REQUIRED)
  find_package(glfw3 REQUIRED)
  find_package(OpenGL REQUIRED)
  include_directories(
    /usr/local/Cellar/librealsense/2.29.0/include 
    /usr/local/Cellar/glfw/3.3/include
    ../../third-party/glfw-imgui/include
    FileAccess.hpp
  )
  add_definitions(-DAPPLE)
endif (APPLE)

#file(GLOB imgui_SRC
#  "../../third-party/glfw-imgui/src/*.h"
#  "../../third-party/glfw-imgui/src/*.hpp"
#  "../../third-party/glfw-imgui/src/*.c"
#  "../../third-party/glfw-imgui/src/*.cpp"
#)

add_executable(VolumetricFusion
  #MainWindow.cpp
  RecordPlaybackMainWindow.cpp
  Recorder.cpp
  CaptureDevice.cpp

  #${imgui_SRC}

  #../../third-party/glfw-imgui/src/context.c
  ##../../third-party/glfw-imgui/src/glfw_config.h
  ##../../third-party/glfw-imgui/src/imconfig.h
  ../../third-party/glfw-imgui/src/imgui-fonts-fontawesome.hpp
  ../../third-party/glfw-imgui/src/imgui-fonts-karla.hpp
  ../../third-party/glfw-imgui/src/imgui.cpp
  #../../third-party/glfw-imgui/src/imgui.h
  ../../third-party/glfw-imgui/src/imgui_draw.cpp
  ../../third-party/glfw-imgui/src/imgui_impl_glfw.cpp
  #../../third-party/glfw-imgui/src/imgui_impl_glfw.h
  ##../../third-party/glfw-imgui/src/imgui_internal.h
  #../../third-party/glfw-imgui/src/init.c
  #../../third-party/glfw-imgui/src/input.c
  ##../../third-party/glfw-imgui/src/internal.h
  #../../third-party/glfw-imgui/src/monitor.c
  ##../../third-party/glfw-imgui/src/stb_rect_pack.h
  ##../../third-party/glfw-imgui/src/stb_textedit.h
  ##../../third-party/glfw-imgui/src/stb_truetype.h
  #../../third-party/glfw-imgui/src/wgl_context.c
  ##../../third-party/glfw-imgui/src/wgl_context.h
  #../../third-party/glfw-imgui/src/win32_init.c
  #../../third-party/glfw-imgui/src/win32_monitor.c
  ##../../third-party/glfw-imgui/src/win32_platform.h
  #../../third-party/glfw-imgui/src/win32_time.c
  #../../third-party/glfw-imgui/src/win32_tls.c
  ##../../third-party/glfw-imgui/src/win32_tls.h
  #../../third-party/glfw-imgui/src/win32_window.c
  #../../third-party/glfw-imgui/src/window.c
  #../../third-party/glfw-imgui/src/winmm_joystick.c
  ##../../third-party/glfw-imgui/src/winmm_joystick.h

)
message(STATUS ${imgui_SRC})

target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY})
target_link_libraries(${PROJECT_NAME} glfw ${GLFW_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OPENGL_gl_LIBRARY})



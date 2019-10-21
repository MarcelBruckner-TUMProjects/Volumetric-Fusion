cmake_minimum_required(VERSION 3.12.2)

project(hello-realsense)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release"
    CACHE STRING "Configuration types" FORCE)

add_executable(hello-realsense
  rs-hello-realsense.cpp
)
# Preprocessor definitions
target_compile_definitions(hello-realsense PRIVATE
  $<$<CONFIG:Debug>:_MBCS;>
  $<$<CONFIG:Release>:_MBCS;>
)

# SDL check
target_compile_options(hello-realsense PRIVATE
  "$<$<CONFIG:Debug>:/sdl>"
  "$<$<CONFIG:Release>:/sdl>"
)

# Minimal rebuild
if (MSVC)
  target_compile_options(hello-realsense PRIVATE
    "$<$<CONFIG:Debug>:/Gm->"
    "$<$<CONFIG:Release>:/Gm->"
  )
endif ()

# Precompiled header files
if (MSVC)
  target_compile_options(hello-realsense PRIVATE
    ""
  )
endif ()

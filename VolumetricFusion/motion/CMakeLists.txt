cmake_minimum_required(VERSION 3.12.2)

project(motion)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release"
    CACHE STRING "Configuration types" FORCE)

add_executable(motion
  ../../third-party/lz4/lz4.c
  ../../third-party/lz4/lz4.h
  ../d435.h
  ../example.hpp
  rs-motion.cpp
)
# Additional include directories
set_property(TARGET motion
  APPEND PROPERTY INCLUDE_DIRECTORIES
  $<$<CONFIG:Debug>:
    ${CMAKE_SOURCE_DIR};
    ../..\third-party/lz4>
  $<$<CONFIG:Release>:
    ${CMAKE_SOURCE_DIR};
    ../../third-party/lz4>
)

# Preprocessor definitions
target_compile_definitions(motion PRIVATE
  $<$<CONFIG:Debug>:_MBCS;_CRT_SECURE_NO_WARNINGS>
  $<$<CONFIG:Release>:_MBCS;_CRT_SECURE_NO_WARNINGS>
)

# SDL check
target_compile_options(motion PRIVATE
  "$<$<CONFIG:Debug>:/sdl>"
  "$<$<CONFIG:Release>:/sdl>"
)

# Minimal rebuild
if (MSVC)
  target_compile_options(motion PRIVATE
    "$<$<CONFIG:Debug>:/Gm->"
    "$<$<CONFIG:Release>:/Gm->"
  )
endif ()

# Precompiled header files
if (MSVC)
  target_compile_options(motion PRIVATE
    ""
  )
endif ()

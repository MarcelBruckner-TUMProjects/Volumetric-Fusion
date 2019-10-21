cmake_minimum_required(VERSION 3.12.2)

project(save-to-disk)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release"
    CACHE STRING "Configuration types" FORCE)

add_executable(save-to-disk
  rs-save-to-disk.cpp
)
# Additional include directories
set_property(TARGET save-to-disk
  APPEND PROPERTY INCLUDE_DIRECTORIES
  $<$<CONFIG:Debug>:
    ${CMAKE_SOURCE_DIR}>
  $<$<CONFIG:Release>:
    ${CMAKE_SOURCE_DIR}>
)

# Preprocessor definitions
target_compile_definitions(save-to-disk PRIVATE
  $<$<CONFIG:Debug>:_MBCS;_CRT_SECURE_NO_WARNINGS>
  $<$<CONFIG:Release>:_MBCS;_CRT_SECURE_NO_WARNINGS>
)

# SDL check
target_compile_options(save-to-disk PRIVATE
  "$<$<CONFIG:Debug>:/sdl>"
  "$<$<CONFIG:Release>:/sdl>"
)

# Minimal rebuild
if (MSVC)
  target_compile_options(save-to-disk PRIVATE
    "$<$<CONFIG:Debug>:/Gm->"
    "$<$<CONFIG:Release>:/Gm->"
  )
endif ()

# Precompiled header files
if (MSVC)
  target_compile_options(save-to-disk PRIVATE
    ""
  )
endif ()

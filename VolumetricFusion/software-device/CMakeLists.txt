cmake_minimum_required(VERSION 3.12.2)

project(software-device)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release"
    CACHE STRING "Configuration types" FORCE)

add_executable(software-device
  rs-software-device.cpp
)
# Additional include directories
set_property(TARGET software-device
  APPEND PROPERTY INCLUDE_DIRECTORIES
  $<$<CONFIG:Debug>:
    ${CMAKE_SOURCE_DIR}>
  $<$<CONFIG:Release>:
    ${CMAKE_SOURCE_DIR}>
)

# Preprocessor definitions
target_compile_definitions(software-device PRIVATE
  $<$<CONFIG:Debug>:_MBCS;_CRT_SECURE_NO_WARNINGS>
  $<$<CONFIG:Release>:_MBCS;_CRT_SECURE_NO_WARNINGS>
)

# SDL check
target_compile_options(software-device PRIVATE
  "$<$<CONFIG:Debug>:/sdl>"
  "$<$<CONFIG:Release>:/sdl>"
)

# Minimal rebuild
if (MSVC)
  target_compile_options(software-device PRIVATE
    "$<$<CONFIG:Debug>:/Gm->"
    "$<$<CONFIG:Release>:/Gm->"
  )
endif ()

# Precompiled header files
if (MSVC)
  target_compile_options(software-device PRIVATE
    ""
  )
endif ()

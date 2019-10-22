# Install script for directory: /Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/align/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/align-advanced/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/ar-basic/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/capture/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/measure/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/motion/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/multicam/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/pointcloud/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/pose/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/pose-predict/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/post-processing/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/record-playback/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/save-to-disk/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/sensor-control/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/trajectory/cmake_install.cmake")
  include("/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/VolumetricFusion/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")

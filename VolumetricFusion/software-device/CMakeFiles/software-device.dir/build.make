# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.15.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.15.4/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion"

# Include any dependencies generated for this target.
include software-device/CMakeFiles/software-device.dir/depend.make

# Include the progress variables for this target.
include software-device/CMakeFiles/software-device.dir/progress.make

# Include the compile flags for this target's objects.
include software-device/CMakeFiles/software-device.dir/flags.make

software-device/CMakeFiles/software-device.dir/rs-software-device.cpp.o: software-device/CMakeFiles/software-device.dir/flags.make
software-device/CMakeFiles/software-device.dir/rs-software-device.cpp.o: software-device/rs-software-device.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object software-device/CMakeFiles/software-device.dir/rs-software-device.cpp.o"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/software-device.dir/rs-software-device.cpp.o -c "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device/rs-software-device.cpp"

software-device/CMakeFiles/software-device.dir/rs-software-device.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/software-device.dir/rs-software-device.cpp.i"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device/rs-software-device.cpp" > CMakeFiles/software-device.dir/rs-software-device.cpp.i

software-device/CMakeFiles/software-device.dir/rs-software-device.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/software-device.dir/rs-software-device.cpp.s"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device/rs-software-device.cpp" -o CMakeFiles/software-device.dir/rs-software-device.cpp.s

# Object files for target software-device
software__device_OBJECTS = \
"CMakeFiles/software-device.dir/rs-software-device.cpp.o"

# External object files for target software-device
software__device_EXTERNAL_OBJECTS =

software-device/software-device: software-device/CMakeFiles/software-device.dir/rs-software-device.cpp.o
software-device/software-device: software-device/CMakeFiles/software-device.dir/build.make
software-device/software-device: software-device/CMakeFiles/software-device.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable software-device"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/software-device.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
software-device/CMakeFiles/software-device.dir/build: software-device/software-device

.PHONY : software-device/CMakeFiles/software-device.dir/build

software-device/CMakeFiles/software-device.dir/clean:
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device" && $(CMAKE_COMMAND) -P CMakeFiles/software-device.dir/cmake_clean.cmake
.PHONY : software-device/CMakeFiles/software-device.dir/clean

software-device/CMakeFiles/software-device.dir/depend:
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/software-device/CMakeFiles/software-device.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : software-device/CMakeFiles/software-device.dir/depend


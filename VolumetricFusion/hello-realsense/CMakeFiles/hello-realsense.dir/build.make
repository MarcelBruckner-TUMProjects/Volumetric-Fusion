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
include hello-realsense/CMakeFiles/hello-realsense.dir/depend.make

# Include the progress variables for this target.
include hello-realsense/CMakeFiles/hello-realsense.dir/progress.make

# Include the compile flags for this target's objects.
include hello-realsense/CMakeFiles/hello-realsense.dir/flags.make

hello-realsense/CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.o: hello-realsense/CMakeFiles/hello-realsense.dir/flags.make
hello-realsense/CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.o: hello-realsense/rs-hello-realsense.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object hello-realsense/CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.o"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.o -c "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense/rs-hello-realsense.cpp"

hello-realsense/CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.i"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense/rs-hello-realsense.cpp" > CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.i

hello-realsense/CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.s"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense/rs-hello-realsense.cpp" -o CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.s

# Object files for target hello-realsense
hello__realsense_OBJECTS = \
"CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.o"

# External object files for target hello-realsense
hello__realsense_EXTERNAL_OBJECTS =

hello-realsense/hello-realsense: hello-realsense/CMakeFiles/hello-realsense.dir/rs-hello-realsense.cpp.o
hello-realsense/hello-realsense: hello-realsense/CMakeFiles/hello-realsense.dir/build.make
hello-realsense/hello-realsense: hello-realsense/CMakeFiles/hello-realsense.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hello-realsense"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello-realsense.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
hello-realsense/CMakeFiles/hello-realsense.dir/build: hello-realsense/hello-realsense

.PHONY : hello-realsense/CMakeFiles/hello-realsense.dir/build

hello-realsense/CMakeFiles/hello-realsense.dir/clean:
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense" && $(CMAKE_COMMAND) -P CMakeFiles/hello-realsense.dir/cmake_clean.cmake
.PHONY : hello-realsense/CMakeFiles/hello-realsense.dir/clean

hello-realsense/CMakeFiles/hello-realsense.dir/depend:
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/hello-realsense/CMakeFiles/hello-realsense.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : hello-realsense/CMakeFiles/hello-realsense.dir/depend


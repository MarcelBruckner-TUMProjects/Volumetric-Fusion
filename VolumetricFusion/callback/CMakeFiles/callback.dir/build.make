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
include callback/CMakeFiles/callback.dir/depend.make

# Include the progress variables for this target.
include callback/CMakeFiles/callback.dir/progress.make

# Include the compile flags for this target's objects.
include callback/CMakeFiles/callback.dir/flags.make

callback/CMakeFiles/callback.dir/rs-callback.cpp.o: callback/CMakeFiles/callback.dir/flags.make
callback/CMakeFiles/callback.dir/rs-callback.cpp.o: callback/rs-callback.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object callback/CMakeFiles/callback.dir/rs-callback.cpp.o"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/callback.dir/rs-callback.cpp.o -c "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback/rs-callback.cpp"

callback/CMakeFiles/callback.dir/rs-callback.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/callback.dir/rs-callback.cpp.i"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback/rs-callback.cpp" > CMakeFiles/callback.dir/rs-callback.cpp.i

callback/CMakeFiles/callback.dir/rs-callback.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/callback.dir/rs-callback.cpp.s"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback/rs-callback.cpp" -o CMakeFiles/callback.dir/rs-callback.cpp.s

# Object files for target callback
callback_OBJECTS = \
"CMakeFiles/callback.dir/rs-callback.cpp.o"

# External object files for target callback
callback_EXTERNAL_OBJECTS =

callback/callback: callback/CMakeFiles/callback.dir/rs-callback.cpp.o
callback/callback: callback/CMakeFiles/callback.dir/build.make
callback/callback: callback/CMakeFiles/callback.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable callback"
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/callback.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
callback/CMakeFiles/callback.dir/build: callback/callback

.PHONY : callback/CMakeFiles/callback.dir/build

callback/CMakeFiles/callback.dir/clean:
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback" && $(CMAKE_COMMAND) -P CMakeFiles/callback.dir/cmake_clean.cmake
.PHONY : callback/CMakeFiles/callback.dir/clean

callback/CMakeFiles/callback.dir/depend:
	cd "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback" "/Users/lilith/Google Drive/Uni TUM/Practical 3D Scanning & Spatial Learning/VolumetricFusion/callback/CMakeFiles/callback.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : callback/CMakeFiles/callback.dir/depend


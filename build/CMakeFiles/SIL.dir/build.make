# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/qzx/.local/lib/python3.6/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/qzx/.local/lib/python3.6/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/qzx/code/SIL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qzx/code/SIL/build

# Include any dependencies generated for this target.
include CMakeFiles/SIL.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SIL.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SIL.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SIL.dir/flags.make

CMakeFiles/SIL.dir/src/runable.cpp.o: CMakeFiles/SIL.dir/flags.make
CMakeFiles/SIL.dir/src/runable.cpp.o: ../src/runable.cpp
CMakeFiles/SIL.dir/src/runable.cpp.o: CMakeFiles/SIL.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qzx/code/SIL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SIL.dir/src/runable.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SIL.dir/src/runable.cpp.o -MF CMakeFiles/SIL.dir/src/runable.cpp.o.d -o CMakeFiles/SIL.dir/src/runable.cpp.o -c /home/qzx/code/SIL/src/runable.cpp

CMakeFiles/SIL.dir/src/runable.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SIL.dir/src/runable.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qzx/code/SIL/src/runable.cpp > CMakeFiles/SIL.dir/src/runable.cpp.i

CMakeFiles/SIL.dir/src/runable.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SIL.dir/src/runable.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qzx/code/SIL/src/runable.cpp -o CMakeFiles/SIL.dir/src/runable.cpp.s

CMakeFiles/SIL.dir/src/detect/detector.cpp.o: CMakeFiles/SIL.dir/flags.make
CMakeFiles/SIL.dir/src/detect/detector.cpp.o: ../src/detect/detector.cpp
CMakeFiles/SIL.dir/src/detect/detector.cpp.o: CMakeFiles/SIL.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qzx/code/SIL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/SIL.dir/src/detect/detector.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SIL.dir/src/detect/detector.cpp.o -MF CMakeFiles/SIL.dir/src/detect/detector.cpp.o.d -o CMakeFiles/SIL.dir/src/detect/detector.cpp.o -c /home/qzx/code/SIL/src/detect/detector.cpp

CMakeFiles/SIL.dir/src/detect/detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SIL.dir/src/detect/detector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qzx/code/SIL/src/detect/detector.cpp > CMakeFiles/SIL.dir/src/detect/detector.cpp.i

CMakeFiles/SIL.dir/src/detect/detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SIL.dir/src/detect/detector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qzx/code/SIL/src/detect/detector.cpp -o CMakeFiles/SIL.dir/src/detect/detector.cpp.s

CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.o: CMakeFiles/SIL.dir/flags.make
CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.o: ../src/recognise/recogniser.cpp
CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.o: CMakeFiles/SIL.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qzx/code/SIL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.o -MF CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.o.d -o CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.o -c /home/qzx/code/SIL/src/recognise/recogniser.cpp

CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qzx/code/SIL/src/recognise/recogniser.cpp > CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.i

CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qzx/code/SIL/src/recognise/recogniser.cpp -o CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.s

CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.o: CMakeFiles/SIL.dir/flags.make
CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.o: ../src/postprocess/postprocessor.cpp
CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.o: CMakeFiles/SIL.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qzx/code/SIL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.o -MF CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.o.d -o CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.o -c /home/qzx/code/SIL/src/postprocess/postprocessor.cpp

CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qzx/code/SIL/src/postprocess/postprocessor.cpp > CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.i

CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qzx/code/SIL/src/postprocess/postprocessor.cpp -o CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.s

CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.o: CMakeFiles/SIL.dir/flags.make
CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.o: ../src/preprocess/preprocessor.cpp
CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.o: CMakeFiles/SIL.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qzx/code/SIL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.o -MF CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.o.d -o CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.o -c /home/qzx/code/SIL/src/preprocess/preprocessor.cpp

CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qzx/code/SIL/src/preprocess/preprocessor.cpp > CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.i

CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qzx/code/SIL/src/preprocess/preprocessor.cpp -o CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.s

CMakeFiles/SIL.dir/src/visualize/visualization.cpp.o: CMakeFiles/SIL.dir/flags.make
CMakeFiles/SIL.dir/src/visualize/visualization.cpp.o: ../src/visualize/visualization.cpp
CMakeFiles/SIL.dir/src/visualize/visualization.cpp.o: CMakeFiles/SIL.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qzx/code/SIL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/SIL.dir/src/visualize/visualization.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SIL.dir/src/visualize/visualization.cpp.o -MF CMakeFiles/SIL.dir/src/visualize/visualization.cpp.o.d -o CMakeFiles/SIL.dir/src/visualize/visualization.cpp.o -c /home/qzx/code/SIL/src/visualize/visualization.cpp

CMakeFiles/SIL.dir/src/visualize/visualization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SIL.dir/src/visualize/visualization.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qzx/code/SIL/src/visualize/visualization.cpp > CMakeFiles/SIL.dir/src/visualize/visualization.cpp.i

CMakeFiles/SIL.dir/src/visualize/visualization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SIL.dir/src/visualize/visualization.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qzx/code/SIL/src/visualize/visualization.cpp -o CMakeFiles/SIL.dir/src/visualize/visualization.cpp.s

# Object files for target SIL
SIL_OBJECTS = \
"CMakeFiles/SIL.dir/src/runable.cpp.o" \
"CMakeFiles/SIL.dir/src/detect/detector.cpp.o" \
"CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.o" \
"CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.o" \
"CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.o" \
"CMakeFiles/SIL.dir/src/visualize/visualization.cpp.o"

# External object files for target SIL
SIL_EXTERNAL_OBJECTS =

SIL: CMakeFiles/SIL.dir/src/runable.cpp.o
SIL: CMakeFiles/SIL.dir/src/detect/detector.cpp.o
SIL: CMakeFiles/SIL.dir/src/recognise/recogniser.cpp.o
SIL: CMakeFiles/SIL.dir/src/postprocess/postprocessor.cpp.o
SIL: CMakeFiles/SIL.dir/src/preprocess/preprocessor.cpp.o
SIL: CMakeFiles/SIL.dir/src/visualize/visualization.cpp.o
SIL: CMakeFiles/SIL.dir/build.make
SIL: /usr/local/lib/libopencv_dnn.so.3.4.14
SIL: /usr/local/lib/libopencv_highgui.so.3.4.14
SIL: /usr/local/lib/libopencv_ml.so.3.4.14
SIL: /usr/local/lib/libopencv_objdetect.so.3.4.14
SIL: /usr/local/lib/libopencv_shape.so.3.4.14
SIL: /usr/local/lib/libopencv_stitching.so.3.4.14
SIL: /usr/local/lib/libopencv_superres.so.3.4.14
SIL: /usr/local/lib/libopencv_videostab.so.3.4.14
SIL: /usr/local/lib/libopencv_viz.so.3.4.14
SIL: /usr/local/lib/libopencv_calib3d.so.3.4.14
SIL: /usr/local/lib/libopencv_features2d.so.3.4.14
SIL: /usr/local/lib/libopencv_flann.so.3.4.14
SIL: /usr/local/lib/libopencv_photo.so.3.4.14
SIL: /usr/local/lib/libopencv_video.so.3.4.14
SIL: /usr/local/lib/libopencv_videoio.so.3.4.14
SIL: /usr/local/lib/libopencv_imgcodecs.so.3.4.14
SIL: /usr/local/lib/libopencv_imgproc.so.3.4.14
SIL: /usr/local/lib/libopencv_core.so.3.4.14
SIL: CMakeFiles/SIL.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qzx/code/SIL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable SIL"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SIL.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SIL.dir/build: SIL
.PHONY : CMakeFiles/SIL.dir/build

CMakeFiles/SIL.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SIL.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SIL.dir/clean

CMakeFiles/SIL.dir/depend:
	cd /home/qzx/code/SIL/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qzx/code/SIL /home/qzx/code/SIL /home/qzx/code/SIL/build /home/qzx/code/SIL/build /home/qzx/code/SIL/build/CMakeFiles/SIL.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SIL.dir/depend

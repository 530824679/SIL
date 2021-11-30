cmake_minimum_required(VERSION 3.9)
project(SIL)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")                                                                           # 默认是Release模式，设置为Debug才能调试
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)            # 设置可执行文件的输出目录


find_package(OpenCV REQUIRED)

include_directories(
        ./include
        ./include/detect
        ./include/recognise
        ./include/track
        ./include/common
        ./include/preprocess
        ./include/postprocess
        ./include/visualize
        ./include/onnxruntime/cpu
)

link_directories("/home/chenwei/HDD/Project/SIL/lib/cpu")

add_executable(SIL 
        src/main.cpp
        src/runable.cpp
        src/detect/detector.cpp
        src/recognise/recogniser.cpp
        src/postprocess/postprocessor.cpp
        src/preprocess/preprocessor.cpp
        src/visualize/visualization.cpp)

target_link_libraries(SIL ${OpenCV_LIBS} onnxruntime)
cmake_minimum_required(VERSION 3.9)
project(SIL)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)

include_directories(
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

add_executable(SIL src/runable.cpp
        src/detect/detector.cpp
        src/recognise/recogniser.cpp
        src/postprocess/postprocessor.cpp
        src/preprocess/preprocessor.cpp
        src/visualize/visualization.cpp)

target_link_libraries(SIL ${OpenCV_LIBS} onnxruntime)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(LibTorch-ResNet)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/lib/libtorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(CUDA 12.0 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17) # To use C++ <filesystem>


set(Target1 dnn-mnist)
add_executable(${Target1} simple_main_mnist.cpp)
set_target_properties(${Target1} PROPERTIES CXX_STANDARD 14)
target_link_libraries(${Target1} PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(${Target1} PRIVATE ${CMAKE_SOURCE_DIR}/include)

set(Target2 ocv)
add_executable(${Target2} opencv.cpp)
set_target_properties(${Target2} PROPERTIES CXX_STANDARD 14)
target_link_libraries(${Target2} PRIVATE ${OpenCV_LIBS})
target_include_directories(${Target2} PRIVATE ${CMAKE_SOURCE_DIR}/include /opt/homebrew/Cellar/opencv/4.7.0_2/include/opencv4)

install(TARGETS ${Target1} ${Target2} CONFIGURATIONS Release RUNTIME DESTINATION playground/bin)

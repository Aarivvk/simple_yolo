cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet50)

list(APPEND CMAKE_PREFIX_PATH "../libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(resnet50 main.cpp)
target_link_libraries(resnet50 "${TORCH_LIBRARIES}")
target_include_directories(resnet50 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                           $<INSTALL_INTERFACE:include)
set_property(TARGET resnet50 PROPERTY CXX_STANDARD 14)
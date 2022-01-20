cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet50)

list(APPEND CMAKE_PREFIX_PATH "../libtorch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(Target resnet50-mnist)
add_executable(${Target} main.cpp)
set_target_properties(${Target} PROPERTIES CXX_STANDARD 14)
target_link_libraries(${Target} "${TORCH_LIBRARIES}")
target_include_directories(${Target} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                                           $<INSTALL_INTERFACE:include)


install(TARGETS ${Target} CONFIGURATIONS Release RUNTIME DESTINATION playground/bin)
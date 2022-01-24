cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(LibTorch-ResNet)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17) # To use C++ <filesystem>


add_library(cifar10 cifar10.cpp)
set_target_properties(cifar10 PROPERTIES CXX_STANDARD 14)
target_include_directories(cifar10 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cifar10 PRIVATE "${TORCH_LIBRARIES}")


# add_library(resnet resnet-liem.cpp)
# set_target_properties(resnet PROPERTIES CXX_STANDARD 17) 
# target_include_directories(resnet PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(resnet PRIVATE "${TORCH_LIBRARIES}")

# add_executable(resnet-cifar-demo resnet-cifar-demo.cpp)
# set_target_properties(resnet-cifar-demo PROPERTIES CXX_STANDARD 17) # To use C++ <filesystem>
# target_include_directories(resnet-cifar-demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(resnet-cifar-demo PRIVATE cifar10 resnet "${TORCH_LIBRARIES}")

set(Target resnet50-mnist)
add_executable(${Target} main_mnist.cpp)
set_target_properties(${Target} PROPERTIES CXX_STANDARD 14)
target_link_libraries(${Target} PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(${Target} PRIVATE ${CMAKE_SOURCE_DIR}/include)

set(Target1 dnn-mnist)
add_executable(${Target1} simple_main_mnist.cpp)
set_target_properties(${Target1} PROPERTIES CXX_STANDARD 14)
target_link_libraries(${Target1} PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(${Target1} PRIVATE ${CMAKE_SOURCE_DIR}/include)

set(Target2 resnet50-cifar10)
add_executable(${Target2} main_cifar10.cpp)
set_target_properties(${Target2} PROPERTIES CXX_STANDARD 14)
target_link_libraries(${Target2} PRIVATE cifar10 "${TORCH_LIBRARIES}")
target_include_directories(${Target2} PRIVATE ${CMAKE_SOURCE_DIR}/include)


install(TARGETS ${Target} ${Target1} ${Target2} CONFIGURATIONS Release RUNTIME DESTINATION playground/bin)
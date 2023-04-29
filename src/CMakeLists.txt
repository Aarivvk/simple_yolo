cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(YOLOv3)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.3.0
)
FetchContent_MakeAvailable(tomlplusplus)

FetchContent_GetProperties(tomlplusplus)
if(NOT tomlplusplus_POPULATED)   
  FetchContent_Populate(tomlplusplus)
  message(STATUS "tomlplusplus source dir: ${tomlplusplus_SOURCE_DIR}")
  message(STATUS "tomlplusplus binary dir: ${tomlplusplus_BINARY_DIR}")
  add_subdirectory(${tomlplusplus_SOURCE_DIR} ${tomlplusplus_BINARY_DIR}) 
endif()

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/lib/libtorch)
set(OpenCV_Includes "/opt/homebrew/Cellar/opencv/4.7.0_2/include/opencv4")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

#find_package(CUDA 12.0 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17) # To use C++ <filesystem>

set(Target yolo-dataset)
add_library(${Target} yolo-dataset.cpp)
target_link_libraries(${Target} PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES})
target_include_directories(${Target} PRIVATE ${CMAKE_SOURCE_DIR}/include ${OpenCV_Includes})

set(Target4 yolo-model)
add_library(${Target4} yolo-model.cpp)
message(STATUS "tomlplusplus include  ${tomlplusplus_SOURCE_DIR}/include")
target_link_libraries(${Target4} PRIVATE ${TORCH_LIBRARIES} ${tomlplusplus_INCLUDE_DIR})
target_include_directories(${Target4} PRIVATE ${CMAKE_SOURCE_DIR}/include ${tomlplusplus_SOURCE_DIR}/include)

set(Target3 yolo-main)
add_executable(${Target3} yolo-main.cpp)
target_link_libraries(${Target3} PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES} yolo-dataset yolo-model)
target_include_directories(${Target3} PRIVATE ${CMAKE_SOURCE_DIR}/include ${OpenCV_Includes})


install(TARGETS ${Target} ${Target3} ${Target4} CONFIGURATIONS Release RUNTIME DESTINATION playground/bin)

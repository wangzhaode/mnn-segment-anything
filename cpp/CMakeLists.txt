cmake_minimum_required(VERSION 3.0)
project(sam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# libs dir
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

# target
add_executable(sam_demo ${SRCS})

# link
if (MSVC)
    target_link_libraries(sam_demo MNN)
else()
    target_link_libraries(sam_demo MNN MNN_Express MNNOpenCV log)
endif()

project(cuPMP)
project(cuPMP LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.10)

# set output paths
set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

# set default compiler flags
if((UNIX OR APPLE) AND CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pedantic -Wextra -Wno-long-long")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pedantic -Wextra -Wno-long-long")
endif()

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES")
endif()

add_subdirectory(src/TriMesh)
#add_subdirectory(doc)
#add_subdirectory(examples)
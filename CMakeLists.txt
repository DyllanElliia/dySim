cmake_minimum_required(VERSION 3.10)
# program name
project(dyMath)

# add package
find_package(Threads)
set(dyMath_libs ${CMAKE_THREAD_LIBS_INIT})

# set compiler argument
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/src)

# read main.cpp
file(GLOB MAIN "src/*.cpp")

# read other file
file(GLOB SOURCE
      "src/*.h"
      "src/*.hpp"
      "src/*.vs"
      "src/*.fs"
      "src/*.gs"
)

# target
FILE(GLOB_RECURSE SRC_FILES "*.c" "*.cpp")
foreach(MAIN_PATH ${MAIN})
  string(REGEX REPLACE ".+/(.+)\\..*" "\\1" MAIN_NAME ${MAIN_PATH})
  message(STATUS "build ${MAIN_PATH} -> ${MAIN_NAME}")
  add_executable(${MAIN_NAME} ${MAIN_PATH} ${SOURCE})
  target_link_libraries(${MAIN_NAME} ${CMAKE_THREAD_LIBS_INIT} ${dyMath_libs})
endforeach(MAIN_PATH)

# include file
include_directories(${PROJECT_SOURCE_DIR}/includes)
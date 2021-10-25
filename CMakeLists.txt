cmake_minimum_required(VERSION 3.10)
# program name
project(dyMath)

# add package
find_package(OpenCL REQUIRED)
find_package(Threads)
set(dyMath_libs ${dyMath_libs} ${CMAKE_THREAD_LIBS_INIT} OpenCL::OpenCL)

# !!! link opengl (copy from the source of LearnOpenGL)
# Begin:
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
link_directories(${PROJECT_SOURCE_DIR}/lib)

# find the required packages
find_package(GLM REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")
find_package(GLFW3 REQUIRED)
message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")
find_package(ASSIMP REQUIRED)
message(STATUS "Found ASSIMP in ${ASSIMP_INCLUDE_DIR}")
# find_package(SOIL REQUIRED)
# message(STATUS "Found SOIL in ${SOIL_INCLUDE_DIR}")
# find_package(GLEW REQUIRED)
# message(STATUS "Found GLEW in ${GLEW_INCLUDE_DIR}")

if (POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY)
endif()

if(WIN32)
  set(LIBS glfw3 opengl32 assimp freetype irrKlang)
elseif(UNIX AND NOT APPLE)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  find_package(OpenGL REQUIRED)
  add_definitions(${OPENGL_DEFINITIONS})
  find_package(X11 REQUIRED)
  # note that the order is important for setting the libs
  # use pkg-config --libs $(pkg-config --print-requires --print-requires-private glfw3) in a terminal to confirm
  set(LIBS ${GLFW3_LIBRARY} X11 Xrandr Xinerama Xi Xxf86vm Xcursor GL dl pthread freetype ${ASSIMP_LIBRARY})
  set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")
elseif(APPLE)
  INCLUDE_DIRECTORIES(/System/Library/Frameworks)
  FIND_LIBRARY(COCOA_LIBRARY Cocoa)
  FIND_LIBRARY(OpenGL_LIBRARY OpenGL)
  FIND_LIBRARY(IOKit_LIBRARY IOKit)
  FIND_LIBRARY(CoreVideo_LIBRARY CoreVideo)
  MARK_AS_ADVANCED(COCOA_LIBRARY OpenGL_LIBRARY)
  SET(APPLE_LIBS ${COCOA_LIBRARY} ${IOKit_LIBRARY} ${OpenGL_LIBRARY} ${CoreVideo_LIBRARY})
  SET(APPLE_LIBS ${APPLE_LIBS} ${GLFW3_LIBRARY} ${ASSIMP_LIBRARY})
  set(LIBS ${LIBS} ${APPLE_LIBS})
else()
  set(LIBS )
endif(WIN32)

configure_file(configuration/root_directory.h.in configuration/root_directory.h)
include_directories(${CMAKE_BINARY_DIR}/configuration)

# first create relevant static libraries requried for other projects
# add_library(STB_IMAGE "src/stb_image.cpp")
# set(LIBS ${LIBS} STB_IMAGE)

add_library(GLAD "core/glad.c")
set(LIBS ${LIBS} GLAD)

# !!!End;
# add LIBS to dyMath_libs
set(dyMath_libs ${dyMath_libs} ${LIBS})

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
  string(CONCAT MAIN_NAME ${MAIN_NAME} ".out")
  message(STATUS "-> build ${MAIN_PATH} => ${MAIN_NAME}")
  add_executable(${MAIN_NAME} ${MAIN_PATH} ${SOURCE})
  target_link_libraries(${MAIN_NAME} ${CMAKE_THREAD_LIBS_INIT} ${dyMath_libs})
endforeach(MAIN_PATH)

# include file
include_directories(${PROJECT_SOURCE_DIR}/includes)
include_directories(${PROJECT_SOURCE_DIR}/core)
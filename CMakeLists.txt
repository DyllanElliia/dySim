cmake_minimum_required(VERSION 3.18)
# program name
project(dyMath C CXX)

# set compiler argument
set(CMAKE_BUILD_TYPE Release)

option(CUDA_ENABLE "Use CUDA" ON)
if(CUDA_ENABLE)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
  find_package(CUDA REQUIRED)
  message(STATUS "-> CUDA included at ${CUDA_INCLUDE_DIRS}")
  file(GLOB CUDA_SOURCE "core/src/cuda/*")
  message(STATUS "-> CUDA_SOURCE ${CUDA_SOURCE}")
  add_library(dyMath_cuda STATIC ${CUDA_SOURCE})
  target_compile_features(dyMath_cuda PRIVATE cuda_std_17)
  set_target_properties(dyMath_cuda
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET dyMath_cuda PROPERTY CUDA_ARCHITECTURES 61 70 75)
  set(dyMath_libs ${dyMath_libs} dyMath_cuda)
endif(CUDA_ENABLE)

# add package
# find_package(OpenCL REQUIRED)
find_package(Threads)
# set(dyMath_libs ${dyMath_libs} ${CMAKE_THREAD_LIBS_INIT} OpenCL::OpenCL)
set(dyMath_libs ${dyMath_libs} ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED True)

# !!! link opengl (copy from the source of LearnOpenGL)
# Begin:
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
link_directories(${PROJECT_SOURCE_DIR}/lib)

# find the required packages
find_package(GLM REQUIRED)
message(STATUS "-> GLM included at ${GLM_INCLUDE_DIR}")
find_package(GLFW3 REQUIRED)
message(STATUS "-> GLFW3 included at ${GLFW3_INCLUDE_DIR}")
find_package(ASSIMP REQUIRED)
message(STATUS "-> ASSIMP included at ${ASSIMP_INCLUDE_DIR}")
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
# target_compile_features(GLAD PRIVATE cxx_std_17)
set(LIBS ${LIBS} GLAD)

# !!!End;
# add LIBS to dyMath_libs
# find_package(cuBLAS)
# add_library(dyMathCore "core/src/*.cpp")
set(dyMath_libs ${dyMath_libs} ${LIBS})

# set output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/src)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# read main.cpp
file(GLOB MAIN "src/*_main.cpp" "src/*_main.cu")

# read other file
file(GLOB SOURCE
      "src/*.h"
      "src/*.hpp"
      # "src/*.cpp"
      # "src/*.cu"
      "src/*.vs"
      "src/*.fs"
      "src/*.gs"
)

# target
FILE(GLOB_RECURSE SRC_FILES "*.c" "*.cpp" "*.cu")
foreach(MAIN_PATH ${MAIN})
  string(REGEX REPLACE ".+/(.+)\\..*" "\\1" MAIN_NAME ${MAIN_PATH})
  string(CONCAT MAIN_NAME ${MAIN_NAME} ".out")
  message(STATUS "-> build ${MAIN_PATH} => ${MAIN_NAME}")
  add_executable(${MAIN_NAME} ${MAIN_PATH} ${SOURCE})
  if(CUDA_ENABLE)
    set_property(TARGET ${MAIN_NAME}
                  PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${MAIN_NAME} PROPERTY CUDA_ARCHITECTURES 61 70 75)
    # target_compile_features(dyMath_cuda PRIVATE cuda_std_17 cxx_std_20)
  endif(CUDA_ENABLE)
  target_link_libraries(${MAIN_NAME} ${dyMath_libs} )
endforeach(MAIN_PATH)

# include file
include_directories(
  ${PROJECT_SOURCE_DIR}/includes
  ${PROJECT_SOURCE_DIR}/core
  ${GLM_INCLUDE_DIR}
  ${GLFW3_INCLUDE_DIR}
  ${ASSIMP_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  )

cmake_minimum_required(VERSION 3.2)

# program name
project(dyMath C CXX)

# create compile file
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# use LTO
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

option(CUDA_ENABLE "Use CUDA" ON)

if(CUDA_ENABLE)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
  find_package(CUDA REQUIRED)
  message(STATUS "-> CUDA included at ${CUDA_INCLUDE_DIRS}")
  file(GLOB CUDA_SOURCE "core/math/cuda/*")
  message(STATUS "-> CUDA_SOURCE ${CUDA_SOURCE}")
  add_library(dyMath_cuda STATIC ${CUDA_SOURCE})
  target_compile_features(dyMath_cuda PRIVATE cuda_std_17)
  set_target_properties(dyMath_cuda
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
  endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET dyMath_cuda PROPERTY CUDA_ARCHITECTURES 70 75 80)
  set(dyMath_libs ${dyMath_libs} dyMath_cuda)
  add_definitions(-DDYM_USE_CUDA)
endif(CUDA_ENABLE)

# add package
# find_package(OpenCL REQUIRED)
find_package(Threads)

# set(dyMath_libs ${dyMath_libs} ${CMAKE_THREAD_LIBS_INIT} OpenCL::OpenCL)
set(dyMath_libs ${dyMath_libs} ${CMAKE_THREAD_LIBS_INIT})

option(OPENMP_ENABLE "Use OpenMP" ON)

if(OPENMP_ENABLE)
  find_package(OpenMP REQUIRED)

  if(OpenMP_FOUND)
    add_definitions(-DDYM_USE_OPENMP)
    set(dyMath_libs ${dyMath_libs} OpenMP::OpenMP_CXX)
    message(STATUS "-> OpenMP included at ${OpenMP_CXX_INCLUDE_DIRS} ${OpenMP_C_INCLUDE_DIRS}")
  endif(OpenMP_FOUND)
endif(OPENMP_ENABLE)

# set compiler argument
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -w ${OpenMP_CXX_FLAGS}")

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
if(POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY)
endif()

if(WIN32)
  set(LIBS glfw3 opengl32 assimp freetype irrKlang)
elseif(UNIX AND NOT APPLE)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  find_package(OpenGL REQUIRED)
  add_definitions(${OPENGL_DEFINITIONS})
  message(STATUS "-> OpenGL included at ${OPENGL_INCLUDE_DIR}")
  find_package(X11 REQUIRED)
  message(STATUS "-> X11.X11 included at ${X11_X11_INCLUDE_PATH}")

  # note that the order is important for setting the libs
  # use pkg-config --libs $(pkg-config --print-requires --print-requires-private glfw3) in a terminal to confirm
  set(LIBS ${GLFW3_LIBRARY} X11 Xrandr Xinerama Xi Xxf86vm Xcursor GL dl pthread freetype ${ASSIMP_LIBRARY})
  set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")
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
  set(LIBS)
endif(WIN32)

message(STATUS "-> [Fin] Gernate LIBS")
configure_file(configuration/root_directory.h.in configuration/root_directory.h)
include_directories(${CMAKE_BINARY_DIR}/configuration)
message(STATUS "-> [Fin] Gernate configuration")

# first create relevant static libraries requried for other projects
# add_library(STB_IMAGE "src/stb_image.cpp")
# set(LIBS ${LIBS} STB_IMAGE)
add_library(GLAD "core/glad.c")
message(STATUS "-> [Fin] Add library glad")

# target_compile_features(GLAD PRIVATE cxx_std_17)
set(LIBS ${LIBS} GLAD)

# !!!End;
# add LIBS to dyMath_libs
# find_package(cuBLAS)
# add_library(dyMathCore "core/src/*.cpp")
set(dyMath_libs ${dyMath_libs} ${LIBS})
set(dyMath_include_dirs ${dyMath_include_dirs}
  ${PROJECT_SOURCE_DIR}/includes
  ${PROJECT_SOURCE_DIR}/core
  ${GLM_INCLUDE_DIR}
  ${GLFW3_INCLUDE_DIR}
  ${ASSIMP_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${OpenMP_CXX_INCLUDE_DIRS}
  ${OpenMP_C_INCLUDE_DIRS}
)

message(STATUS "-> [Fin] Add libs and include_dirs")

# set output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/src)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# read main.cpp
file(GLOB MAIN
  "src/*_main.cpp"

  "src/*_main.cu"
)

message(STATUS "-> [Fin] Find file: *_main.cpp, *_main.cu")

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

message(STATUS "-> [Fin] Find source file.")

message(STATUS "-> [Run] Targetting file... it wil take a few minutes.")

# target
file(GLOB_RECURSE SRC_FILES "*.c" "*.cpp" "*.cu")

message(STATUS "-> [Fin] Target file.")

foreach(MAIN_PATH ${MAIN})
  string(REGEX REPLACE ".+/(.+)\\..*" "\\1" MAIN_NAME ${MAIN_PATH})
  string(REGEX REPLACE ".+\\.(.+)" "\\1" MAIN_TYPE ${MAIN_PATH})
  string(CONCAT MAIN_NAME ${MAIN_NAME} ".out")
  message(STATUS "==> build ${MAIN_PATH} (Type: ${MAIN_TYPE}) as")
  message(STATUS "    --> ${MAIN_NAME}")
  add_executable(${MAIN_NAME} ${MAIN_PATH} ${SOURCE})

  if(CUDA_ENABLE)
    if(${MAIN_TYPE} STREQUAL "cu")
      message(STATUS "    -> use nvcc")
      set_property(TARGET ${MAIN_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
      set_property(TARGET ${MAIN_NAME} PROPERTY CUDA_ARCHITECTURES 70 75)
    endif()

    # target_compile_features(dyMath_cuda PRIVATE cuda_std_17 cxx_std_20)
  endif(CUDA_ENABLE)

  target_link_libraries(${MAIN_NAME} ${dyMath_libs})
endforeach(MAIN_PATH)

# include file
include_directories(${dyMath_include_dirs})

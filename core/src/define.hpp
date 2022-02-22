/*
 * @Author: DyllanElliia
 * @Date: 2021-11-03 19:04:10
 * @LastEditTime: 2022-01-26 15:33:20
 * @LastEditors: DyllanElliia
 * @Description:
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>
#include <memory>
#include <type_traits>
#include <utility>

#ifdef DYM_USE_OPENMP
#include <omp.h>
#endif

// #define _DYM_USE_CUDA_

#ifdef DYM_USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define _DYM_GENERAL_ __host__ __device__
#define _DYM_DEVICE_ __device__
#define _DYM_GLOBAL_ __global__

#ifdef __clang__
#pragma nv_exec_check_disable
#else
#pragma hd_warning_disable
#endif  // __clang__

#else
#define _DYM_GENERAL_
#define _DYM_DEVICE_
#define _DYM_GLOBAL_

#endif  //_DYM_USE_CUDA_

#define _DYM_LAMBDA_ _DYM_GENERAL_

#define _DYM_ASSERT_(bool_opt, outstr) \
  try {                                \
    if (bool_opt) throw outstr;        \
  } catch (const char *str) {          \
    std::error << str << std::endl;    \
    exit(EXIT_FAILURE);                \
  }

#define _DYM_FORCE_INLINE_ inline __attribute__((always_inline))

typedef float Real;
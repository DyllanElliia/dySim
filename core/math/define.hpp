/*
 * @Author: DyllanElliia
 * @Date: 2021-11-03 19:04:10
 * @LastEditTime: 2022-03-11 16:11:18
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
#include <memory>
#include <numeric>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef DYM_USE_OPENMP
#include <omp.h>
#endif

#include "../tools/str_hash.hpp"
#include "../tools/sugar.hpp"

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
#endif // __clang__

#else
#define _DYM_GENERAL_
#define _DYM_DEVICE_
#define _DYM_GLOBAL_

#endif //_DYM_USE_CUDA_

#define _DYM_LAMBDA_ _DYM_GENERAL_

#define DYM_ERROR(errorString) __DYM_ERROR_CALL(errorString, __FILE__, __LINE__)

#define DYM_ERROR_cs(className, errorString)                                   \
  __DYM_ERROR_CALL(std::string(className) + " Error: " + errorString,          \
                   __FILE__, __LINE__)

inline void __DYM_ERROR_CALL(std::string err, const char *file,
                             const int line) {
  qp_ctrl(tColor::RED, tType::BOLD, tType::UNDERLINE);
  qprint(err, "\n--- error in file <", file, ">, line", line, ".");
  qp_ctrl();
}

#define DYM_WARNING(str) __DYM_WARNING_CALL(str, __FILE__, __LINE__)
#define DYM_WARNING_cs(className, wString)                                     \
  __DYM_WARNING_CALL(std::string(className) + " Warning: " + wString,          \
                     __FILE__, __LINE__)

inline void __DYM_WARNING_CALL(std::string err, const char *file,
                               const int line) {
  qp_ctrl(tColor::YELLOW, tType::BOLD, tType::UNDERLINE);
  qprint(err, "\n--- warning in file <", file, ">, line", line, ".");
  qp_ctrl();
}

#define _DYM_ASSERT_(bool_opt, outstr)                                         \
  try {                                                                        \
    if (bool_opt)                                                              \
      throw outstr;                                                            \
  } catch (const char *str) {                                                  \
    DYM_ERROR(str);                                                            \
    exit(EXIT_FAILURE);                                                        \
  }

#define _DYM_FORCE_INLINE_ inline __attribute__((always_inline))

typedef float lReal;
typedef double Real;
typedef int Reali;
typedef unsigned int uReali;

namespace dym {
const Real Pi = 3.1415926535897932385;
}
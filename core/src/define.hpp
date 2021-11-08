/*
 * @Author: DyllanElliia
 * @Date: 2021-11-03 19:04:10
 * @LastEditTime: 2021-11-08 14:32:55
 * @LastEditors: Please set LastEditors
 * @Description:
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#define _DYM_USE_CUDA_

#ifdef _DYM_USE_CUDA_
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
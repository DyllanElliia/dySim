/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 14:19:52
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-15 15:28:37
 * @Description:
 */
#pragma once

#include "Index.hpp"
#include <algorithm>

#define _DYM_THREAD_CPU_DEFAULT_ 0
#define _DYM_THREAD_CPU_OPENMP_ 1
#define _DYM_THREAD_CPU_CPPSTD_ 2

#ifndef DYM_DEFAULT_THREAD
#define DYM_DEFAULT_THREAD _DYM_THREAD_CPU_OPENMP_
#endif
namespace dym {
namespace {
bool use_thread = true;
}
namespace launch {}
template <class Func>
void Launch(Func &fun, const int begin_i, const int end_i,
            const unsigned short use_thread_type = DYM_DEFAULT_THREAD) {
  if (use_thread) {
    use_thread = false;
    switch (use_thread_type) {
    case _DYM_THREAD_CPU_DEFAULT_: {
      const unsigned int t_num = std::thread::hardware_concurrency() / 3;
      const unsigned int t_step = (end_i - begin_i + t_num) / t_num;
      std::vector<std::thread> t_pool;
      for (unsigned int i = 0; i < t_num; ++i) {
        unsigned int ib = i * t_step + begin_i, ie = (i + 1) * t_step + begin_i;
        if (ie > end_i)
          ie = end_i;
        t_pool.push_back(std::thread(fun, ib, ie));
      }
      std::for_each(t_pool.begin(), t_pool.end(),
                    [](std::thread &t) { t.join(); });
      break;
    }
    case _DYM_THREAD_CPU_OPENMP_: {
#ifdef DYM_USE_OPENMP
      int step = (end_i - begin_i) / 100;
      step = step > 1 ? step : 1;
#pragma omp parallel for
      for (unsigned int i = begin_i; i < end_i; i += step) {
        const auto fend = i + step;
        fun(i, fend < end_i ? fend : end_i);
      }
#else qp_ctrl(tColor::RED, tType::BOLD, tType::UNDERLINE);
      qprint("Launch ERROR: Fault to Use OpenMP!");
      qp_ctrl();
#endif // DYM_USE_OPENMP
      break;
    }
    // case _DYM_THREAD_CPU_CPPSTD_: {
    //   std::for_each(std::execution::par_unseq, begin_i, end_i, ) break;
    // }
    default: {
      qp_ctrl(tColor::RED, tType::BOLD, tType::UNDERLINE);
      qprint("Launch ERROR: Fault to Find ThreadOption!");
      qp_ctrl();
      break;
    }
    }

    use_thread = true;
  } else
    fun(begin_i, end_i);
}

namespace detail {
template <class T, T... inds, class F>
constexpr _DYM_GENERAL_ _DYM_FORCE_INLINE_ void
loop(std::integer_sequence<T, inds...>, F &&f) {
  (f(std::integral_constant<T, inds>{}), ...);
}

} // namespace detail

template <class T, T count, class F>
constexpr _DYM_GENERAL_ _DYM_FORCE_INLINE_ void Loop(F &&f) {
  detail::loop(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

} // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 14:19:52
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-11-23 14:20:54
 * @Description:
 */
#pragma once

#include "Index.hpp"

namespace dym {
namespace {
bool use_thread = true;
}
template <class Func>
void launch(Func &fun, const int begin_i, const int end_i) {
  if (use_thread) {
    use_thread = false;
    const unsigned int t_num = std::thread::hardware_concurrency() / 3;
    const unsigned int t_step = (end_i - begin_i + t_num) / t_num;
    std::vector<std::thread> t_pool;
    for (unsigned int i = 0; i < t_num; ++i) {
      unsigned int ib = i * t_step + begin_i, ie = (i + 1) * t_step + begin_i;
      if (ie > end_i) ie = end_i;
      t_pool.push_back(std::thread(fun, ib, ie));
    }
    std::for_each(t_pool.begin(), t_pool.end(),
                  [](std::thread &t) { t.join(); });
    use_thread = true;
  } else
    fun(begin_i, end_i);
}
}  // namespace dym
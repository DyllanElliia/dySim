/*
 * @Author: DyllanElliia
 * @Date: 2022-01-26 15:16:30
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-27 15:16:39
 * @Description:
 */
#pragma once
#include "Index.hpp"
namespace dym {
namespace real {
_DYM_FORCE_INLINE_ Real clamp(const Real &v, const Real &min_v,
                              const Real &max_v) {
  return v < min_v ? min_v : (v > max_v ? max_v : v);
}
_DYM_FORCE_INLINE_ Real sqr(const Real &v) { return v * v; }
}  // namespace real
}  // namespace dym
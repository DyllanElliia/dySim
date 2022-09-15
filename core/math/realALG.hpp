/*
 * @Author: DyllanElliia
 * @Date: 2022-01-26 15:16:30
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-11 15:35:33
 * @Description:
 */
#pragma once
#include "Index.hpp"

namespace dym {
// namespace real {
template <typename Type, typename TypeS>
_DYM_FORCE_INLINE_ Type clamp(const Type &v, const TypeS &min_v,
                              const TypeS &max_v) {
  return v < (Type)min_v ? min_v : (v > (Type)max_v ? max_v : v);
}
template <typename Type> _DYM_FORCE_INLINE_ Type sqr(const Type &v) {
  return v * v;
}
template <typename Type>
_DYM_FORCE_INLINE_ Type pow(const Type &v, const Type &s) {
  return std::pow(v, s);
}

template <typename Type, typename TypeS>
_DYM_FORCE_INLINE_ Type min(const Type &v1, const TypeS &v2) {
  return v1 < v2 ? v1 : v2;
}
template <typename Type, typename TypeS>
_DYM_FORCE_INLINE_ Type max(const Type &v1, const TypeS &v2) {
  return v1 > v2 ? v1 : v2;
}

template <typename Type> _DYM_FORCE_INLINE_ Type abs(const Type &v) {
  return v > 0 ? v : -v;
}

#define _dym_real_use_std_(fun)                                                \
  template <typename Type>                                                     \
  constexpr _DYM_FORCE_INLINE_ Type fun(const Type &v) {                       \
    return std::fun(v);                                                        \
  }
_dym_real_use_std_(sqrt);
_dym_real_use_std_(cos);
_dym_real_use_std_(cosh);
_dym_real_use_std_(acos);
_dym_real_use_std_(acosh);
_dym_real_use_std_(sin);
_dym_real_use_std_(sinh);
_dym_real_use_std_(asin);
_dym_real_use_std_(asinh);
_dym_real_use_std_(tan);
_dym_real_use_std_(tanh);
_dym_real_use_std_(atan);
_dym_real_use_std_(atanh);
_dym_real_use_std_(cbrt);
_dym_real_use_std_(exp);
_dym_real_use_std_(exp2);
_dym_real_use_std_(expm1);
_dym_real_use_std_(ceil);
_dym_real_use_std_(floor);
_dym_real_use_std_(round);
_dym_real_use_std_(trunc);
_dym_real_use_std_(isinf);
_dym_real_use_std_(isnan);
_dym_real_use_std_(log);
_dym_real_use_std_(log2);
_dym_real_use_std_(log10);
_dym_real_use_std_(log1p);
_dym_real_use_std_(logb);

template <typename Type>
_DYM_FORCE_INLINE_ Type lerp(const Type &v0, const Type &v1, const Real &t) {
  return (1 - t) * v0 + t * v1;
}

// }  // namespace real
} // namespace dym
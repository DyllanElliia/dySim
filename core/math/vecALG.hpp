/*
 * @Author: DyllanElliia
 * @Date: 2022-01-26 15:14:37
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-08 16:56:43
 * @Description:
 */
#pragma once
#include "vector.hpp"
namespace dym {
// namespace vector {
template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Vector<Type, dim> clamp(const Vector<Type, dim>& v,
                                           const Vector<Type, dim>& min_v,
                                           const Vector<Type, dim>& max_v) {
  return Vector<Type, dim>(
      [&](Type& e, int i) { e = clamp(v[i], min_v[i], max_v[i]); });
}
template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Vector<Type, dim> pow(const Vector<Type, dim>& v,
                                         const int& s) {
  return Vector<Type, dim>([&](Type& e, int i) { e = pow(v[i], s); });
}
template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Vector<Type, dim> min(const Vector<Type, dim>& v1,
                                         const Vector<Type, dim>& v2) {
  return Vector<Type, dim>([&](Type& e, int i) { e = min(v1[i], v2[i]); });
}
template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Vector<Type, dim> max(const Vector<Type, dim>& v1,
                                         const Vector<Type, dim>& v2) {
  return Vector<Type, dim>([&](Type& e, int i) { e = max(v1[i], v2[i]); });
}

#define _dym_vector_use_std_(fun)                                        \
  template <typename Type, std::size_t dim>                              \
  _DYM_FORCE_INLINE_ Vector<Type, dim> fun(const Vector<Type, dim>& a) { \
    return Vector<Type, dim>([&](Type& e, int i) { e = fun(a[i]); });    \
  }
_dym_vector_use_std_(sqr);
_dym_vector_use_std_(abs);
_dym_vector_use_std_(sqrt);
_dym_vector_use_std_(cos);
_dym_vector_use_std_(cosh);
_dym_vector_use_std_(acos);
_dym_vector_use_std_(acosh);
_dym_vector_use_std_(sin);
_dym_vector_use_std_(sinh);
_dym_vector_use_std_(asin);
_dym_vector_use_std_(asinh);
_dym_vector_use_std_(tan);
_dym_vector_use_std_(tanh);
_dym_vector_use_std_(atan);
_dym_vector_use_std_(atanh);
_dym_vector_use_std_(cbrt);
_dym_vector_use_std_(exp);
_dym_vector_use_std_(exp2);
_dym_vector_use_std_(expm1);
_dym_vector_use_std_(ceil);
_dym_vector_use_std_(floor);
_dym_vector_use_std_(round);
_dym_vector_use_std_(trunc);
_dym_vector_use_std_(isinf);
_dym_vector_use_std_(isnan);
_dym_vector_use_std_(log);
_dym_vector_use_std_(log2);
_dym_vector_use_std_(log10);
_dym_vector_use_std_(log1p);
_dym_vector_use_std_(logb);

template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Vector<Type, dim> lerp(const Vector<Type, dim>& v0,
                                          const Vector<Type, dim>& v1,
                                          const Real& t) {
  return (1 - t) * v0 + t * v1;
}

template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Vector<Type, dim> Nlerp(const Vector<Type, dim>& v0,
                                           const Vector<Type, dim>& v1,
                                           const Real& t) {
  Vector<Type, dim> r = lerp(v0, v1, t);
  return r.normalize();
}

template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Vector<Type, dim> Slerp(const Vector<Type, dim>& v0,
                                           const Vector<Type, dim>& v1,
                                           const Real& t) {
  Type theta = 1 / cos(v0.dot(v1)), sintheta_inv = 1 / sin(theta);
  return sin((1 - t) * theta) * sintheta_inv * v0 +
         sin(t * theta) * sintheta_inv * v1;
}

// }  // namespace vector
}  // namespace dym
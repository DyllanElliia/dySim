/*
 * @Author: DyllanElliia
 * @Date: 2022-01-26 15:14:37
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-26 15:42:22
 * @Description:
 */
#pragma once
#include "vector.hpp"
namespace dym {
namespace vector {
template <std::size_t dim>
_DYM_FORCE_INLINE_ Vector<Real, dim> sqr(const Vector<Real, dim>& a) {
  return Vector<Real, dim>([&](Real& e, int i) { e = real::sqr(a[i]); });
}
}  // namespace vector
}  // namespace dym
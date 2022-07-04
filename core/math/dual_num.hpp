/*
 * @Author: DyllanElliia
 * @Date: 2022-07-01 15:37:04
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-04 16:24:13
 * @Description:
 */
#pragma once

#include "./matrix.hpp"
#include "math/realALG.hpp"
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <string>
namespace dym {
template <class Type> class DualNum {
private:
  using thisType = DualNum<Type>;

public:
  DualNum(const Type &a, const Type &b) { A = a, B = b; }
  DualNum(const std::initializer_list<Type> &v) {
    _DYM_ASSERT_(v.size() != 2,
                 "DualNum ERROR: please input {A,B} for a + b /epsilon!");
    A = *(v.begin()), B = *(v.begin() + 1);
  }
  DualNum(const thisType &d) { A = d.A, B = d.B; }
  DualNum(const thisType &&d) { A = d.A, B = d.B; }
  DualNum(const Type &vul = 0) { A = vul, B = Type(1); }
  ~DualNum() {}

  inline DualNum operator=(const DualNum &d) {
    A = d.A, B = d.B;
    return *this;
  }
  inline DualNum operator=(const Type &vul) {
    A = vul, B = vul;
    return *this;
  }

  void show() const {
    std::cout << "Dual: [A: " << A << ", B: " << B << +"]" << std::endl;
  }

  friend std::ostream &operator<<(std::ostream &output, const DualNum &d) {
    output << "Dual: [A: " << d.A << ", B: " << d.B << +"]";
    return output;
  }

  _DYM_FORCE_INLINE_ auto conjugate() { return thisType{A, B * Type(-1)}; }
  _DYM_FORCE_INLINE_ auto inverse() {
    return thisType{Type(1) / A, B * Type(-1) / (A * A)};
  }

  thisType operator+(const thisType &rhs) const {
    return {A + rhs.A, B + rhs.B};
  }
  thisType operator-(const thisType &rhs) const {
    return {A - rhs.A, B - rhs.B};
  }
  thisType operator*(const thisType &rhs) const {
    return {A * rhs.A, A * rhs.B + B * rhs.A};
  }
  thisType operator/(const thisType &rhs) const {
    return *this * rhs.inverse();
  }

  thisType &operator+=(const thisType &rhs) {
    A = A + rhs.A, B = B + rhs.B;
    return *this;
  }
  thisType &operator-=(const thisType &rhs) {
    A = A - rhs.A, B = B - rhs.B;
    return *this;
  }
  thisType &operator*=(const thisType &rhs) {
    A = A * rhs.A, B = A * rhs.B + B * rhs.A;
    return *this;
  }
  thisType &operator/=(const thisType &rhs) {
    *this = *this * rhs.inverse();
    return *this;
  }

  friend thisType operator+(const Type &lhs, const thisType &rhs) {
    return {rhs.A + lhs, rhs.B};
  }
  friend thisType operator-(const Type &lhs, const thisType &rhs) {
    return {rhs.A - lhs, rhs.B};
  }
  friend thisType operator*(const Type &lhs, const thisType &rhs) {
    return {rhs.A * lhs, rhs.B * lhs};
  }
  friend thisType operator/(const Type &lhs, const thisType &rhs) {
    return {rhs.A / lhs, rhs.B / lhs};
  }
  friend thisType operator+(const thisType &lhs, const Type &rhs) {
    return {lhs.A + rhs, lhs.B};
  }
  friend thisType operator-(const thisType &lhs, const Type &rhs) {
    return {lhs.A - rhs, lhs.B};
  }
  friend thisType operator*(const thisType &lhs, const Type &rhs) {
    return {lhs.A * rhs, lhs.B * rhs};
  }
  friend thisType operator/(const thisType &lhs, const Type &rhs) {
    return {lhs.A / rhs, lhs.B / rhs};
  }

  thisType &operator+=(const Type &rhs) {
    A = A + rhs, B = B;
    return *this;
  }
  thisType &operator-=(const Type &rhs) {
    A = A - rhs, B = B;
    return *this;
  }
  thisType &operator*=(const Type &rhs) {
    A = A * rhs, B = B * rhs;
    return *this;
  }
  thisType &operator/=(const Type &rhs) {
    A = A / rhs, B = B / rhs;
    return *this;
  }

public:
  Type A, B;
};

template <typename Type, std::size_t m, std::size_t n>
_DYM_FORCE_INLINE_ DualNum<Vector<Type, m>>
operator*(const Matrix<Type, m, n> &ma, const DualNum<Vector<Type, n>> &ve) {
  return {
      Vector<Type, m>([&](Type &e, int i) { e = vector::dot(ma[i], ve.A); }),
      Vector<Type, m>([&](Type &e, int i) { e = vector::dot(ma[i], ve.B); })};
}

#define _dym_dual_num_oneArg_alg_(funname, ...)                                \
  template <typename Type_>                                                    \
  _DYM_FORCE_INLINE_ DualNum<Type_> funname(const DualNum<Type_> &d) {         \
    return __VA_ARGS__;                                                        \
  }

#define _dym_dual_num_twoArg_alg_(funname, argTypeName, ...)                   \
  template <typename Type_>                                                    \
  _DYM_FORCE_INLINE_ DualNum<Type_> funname(const DualNum<Type_> &d,           \
                                            argTypeName) {                     \
    return __VA_ARGS__;                                                        \
  }

_dym_dual_num_oneArg_alg_(sqr, d *d);
_dym_dual_num_twoArg_alg_(pow, const int &s,
                          {dym::pow(d.A, s), s *dym::pow(d.B, s - 1)});
_dym_dual_num_oneArg_alg_(sqrt, {dym::sqrt(d.A), 1 / (2 * dym::sqrt(d.B))});
_dym_dual_num_oneArg_alg_(cos, {dym::cos(d.A), -dym::sin(d.B)});
_dym_dual_num_oneArg_alg_(cosh, {dym::cosh(d.A), dym::sinh(d.B)});
_dym_dual_num_oneArg_alg_(acos,{})
} // namespace dym

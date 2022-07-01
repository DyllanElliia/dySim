/*
 * @Author: DyllanElliia
 * @Date: 2022-07-01 15:37:04
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-01 16:40:07
 * @Description:
 */
#pragma once
#include "math/define.hpp"
#include "realALG.hpp"
#include <cstdlib>
#include <initializer_list>
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
  DualNum(const Type &vul = 0) { A = vul, B = Type(0); }
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

  thisType operator+(const thisType &rhs) { return {A + rhs.A, B + rhs.B}; }
  thisType operator-(const thisType &rhs) { return {A - rhs.A, B - rhs.B}; }
  thisType operator*(const thisType &rhs) {
    return {A * rhs.A, A * rhs.B + B * rhs.A};
  }
  thisType operator/(const thisType &rhs) { return *this * rhs.inverse(); }

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
} // namespace dym
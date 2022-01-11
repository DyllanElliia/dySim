/*
 * @Author: DyllanElliia
 * @Date: 2022-01-07 12:19:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-07 12:40:19
 * @Description:
 */
#pragma once
#include "vector.hpp"

namespace dym {
template <typename Type, int m, int n>
struct Matrix {
 private:
  Vector<Type, m> a[n];

 public:
  Matrix(const Type &num = 0) {
    for (auto &i : a) i = num;
  }
  Matrix(Vector<Type, m>... v) { a = {(v)...}; }
  Matrix(std::function<void(Vector<Type, m> &)> fun) {
    for (auto &e : a) fun(e)
  };
  Matrix(std::function<void(Vector<Type, m> &, int)> fun) {
    int i = 0;
    for (auto &e : a) fun(e, i++);
  };
  Matrix(const Matrix<Type, m, n> &&m) { std::memcpy() }
}
}  // namespace dym

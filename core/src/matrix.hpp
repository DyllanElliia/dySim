/*
 * @Author: DyllanElliia
 * @Date: 2022-01-07 12:19:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-12 16:50:22
 * @Description:
 */
#pragma once
#include "vector.hpp"

namespace dym {
template <typename Type, int m, int n>
struct Matrix {
 private:
  Vector<Type, n> a[m];

 public:
  Matrix(const Type &num = 0) {
    for (auto &i : a) i = num;
  }
  // template <typename... v_args>
  Matrix(const std::vector<std::vector<Type>> &v) {
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) a[i][j] = v[i][j];
  }
  Matrix(std::function<void(Type &)> fun) {
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) fun(a[i][j]);
  };
  Matrix(std::function<void(Type &, int, int)> fun) {
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) fun(a[i][j], i, j);
  };
  Matrix(std::function<void(Vector<Type, n> &)> fun) {
    for (auto &e : a) fun(e);
  };
  Matrix(std::function<void(Vector<Type, n> &, int)> fun) {
    int i = 0;
    for (auto &e : a) fun(e, i++);
  };
  template <int inRank_m, int inRank_n>
  Matrix(const Matrix<Type, inRank_m, inRank_n> &v, const Type &vul = 0) {
    constexpr int for_min = std::min(m, inRank_m);
    for (int i = 0; i < for_min; ++i) a[i] = v[i];
    for (int i = for_min; i < m && i < n; ++i) a[i][i] = vul;
  }
  Matrix(const Matrix<Type, m, n> &&v) { std::memcpy(a, v.a, sizeof(Matrix)); }
  Matrix(const Matrix<Type, m, n> &v) { std::memcpy(a, v.a, sizeof(Matrix)); }

  void show() const {
    std::string res = "Mat: [\n";
    for (auto &v : a) {
      for (int i = 0; i < n; ++i) res += std::to_string(v[i]) + " ";
      res += "\n";
    }
    res += "]";
    std::cout << res << std::endl;
  }
  Vector<Type, n> &operator[](const int &i) { return a[i]; }
  Vector<Type, n> operator[](const int &i) const { return a[i]; }
  template <int inRank_m, int inRank_n>
  Matrix operator=(const Matrix<Type, inRank_m, inRank_n> &v) {
    constexpr int for_min = std::min(m, inRank_m);
    for (int i = 0; i < for_min; ++i) a[i] = v[i];
    return *this;
  }
  Matrix operator=(const Type &num) {
    for (auto &v : a) v = num;
    return *this;
  }
  friend std::ostream &operator<<(std::ostream &output, const Matrix &v) {
    std::string res = "Mat: [\n";
    for (auto &v : v.a) {
      for (int i = 0; i < n; ++i) res += std::to_string(v[i]) + " ";
      res += "\n";
    }
    res += "]\n";
    output << res;
    return output;
  }
  template <typename cType>
  Matrix<cType, m, n> cast() {
    Matrix<cType, m, n> o;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) o[i][j] = a[i][j];
    return o;
  }
};

template <typename Type, int m, int n>
constexpr Vector<Type, m> operator*(const Matrix<Type, m, n> &ma,
                                    const Vector<Type, n> &ve) {
  return Vector<Type, m>([&](Type &e, int i) { e = ma[i] * ve; });
}

}  // namespace dym

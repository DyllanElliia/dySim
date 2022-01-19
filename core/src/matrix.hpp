/*
 * @Author: DyllanElliia
 * @Date: 2022-01-07 12:19:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-19 16:42:02
 * @Description:
 */
#pragma once
#include "vector.hpp"

namespace dym {
template <typename Type, std::size_t m, std::size_t n>
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
  template <std::size_t inRank_m, std::size_t inRank_n>
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
  inline void for_each(std::function<void(Type &)> func) {
    for (auto &e : a) func(e);
  }
  inline void for_each(std::function<void(Type &, int, int)> func) {
    int i = 0;
    for (auto &v : a) v.for_each([&](Type &e, int j) { func(e, i, j); }), i++;
  }

  Vector<Type, n> &operator[](const int &i) { return a[i]; }
  Vector<Type, n> operator[](const int &i) const { return a[i]; }

  template <std::size_t inRank_m, std::size_t inRank_n>
  Matrix operator=(const Matrix<Type, inRank_m, inRank_n> &v) {
    constexpr int for_min = std::min(m, inRank_m);
    for (int i = 0; i < for_min; ++i) a[i] = v[i];
    return *this;
  }
  inline Matrix operator=(const Matrix &v) {
    memcpy(a, v.a, sizeof(Matrix));
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

  constexpr auto shape() const { return gi((int)m, (int)n); }

  inline Matrix<Type, n, m> transpose() const {
    return Matrix<Type, n, m>([&](Type &e, int i, int j) { e = a[j][i]; });
  }
};

template <typename Type, std::size_t m, std::size_t n>
inline Vector<Type, m> operator*(const Matrix<Type, m, n> &ma,
                                 const Vector<Type, n> &ve) {
  return Vector<Type, m>([&](Type &e, int i) { e = ma[i] * ve; });
}

template <typename Type, std::size_t m1, std::size_t n1, std::size_t m2,
          std::size_t n2>
inline Matrix<Type, m1, n2> operator*(const Matrix<Type, m1, n1> &a,
                                      const Matrix<Type, m2, n2> &b) {
  static_assert(n1 == m2,
                "Left Matrix's col must be equal to Right Matrix's row!");
  // dym::matrix::mul_swap
  // mul_fast is better, but difficult to implement. Because of const & ptr
  Matrix<Type, m1, n2> o(0);
  for (int r = 0; r < m1; ++r)
    for (int i = 0; i < n1; ++i)
      for (int c = 0; c < n2; ++c) o[r][c] += a[r][i] * b[i][c];
  return o;
}

#define _dym_matrix_type_operator_binary_(op)                          \
  template <typename Type, std::size_t m, std::size_t n>               \
  inline Matrix<Type, m, n> operator op(const Type &f,                 \
                                        const Matrix<Type, m, n> &s) { \
    return Matrix<Type, m, n>(                                         \
        [&](Type &e, int i, int j) { e = f op s[i][j]; });             \
  }                                                                    \
  template <typename Type, std::size_t m, std::size_t n>               \
  inline Matrix<Type, m, n> operator op(const Matrix<Type, m, n> &f,   \
                                        const Type &s) {               \
    return Matrix<Type, m, n>(                                         \
        [&](Type &e, int i, int j) { e = f[i][j] op s; });             \
  }

_dym_matrix_type_operator_binary_(*);
_dym_matrix_type_operator_binary_(/);

#define _dym_matrix_operator_binary_(op)                               \
  template <typename Type, std::size_t m, std::size_t n>               \
  inline Matrix<Type, m, n> operator op(const Matrix<Type, m, n> &f,   \
                                        const Matrix<Type, m, n> &s) { \
    return Matrix<Type, m, n>(                                         \
        [&](Type &e, int i, int j) { e = f[i][j] op s[i][j]; });       \
  }                                                                    \
  _dym_matrix_type_operator_binary_(op);

#define _dym_matrix_operator_unary_(op)                           \
  template <typename Type, std::size_t m, std::size_t n>          \
  inline void operator op(Matrix<Type, m, n> &f, const Type &s) { \
    for (int i = 0; i < m; ++i)                                   \
      for (int j = 0; j < n; ++j) f[i][j] op s;                   \
  }

_dym_matrix_operator_binary_(+);
_dym_matrix_operator_binary_(-);

_dym_matrix_operator_unary_(+=);
_dym_matrix_operator_unary_(-=);
_dym_matrix_operator_unary_(*=);
_dym_matrix_operator_unary_(/=);

}  // namespace dym

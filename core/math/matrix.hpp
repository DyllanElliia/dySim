/*
 * @Author: DyllanElliia
 * @Date: 2022-01-07 12:19:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-13 16:35:14
 * @Description:
 */
#pragma once
#include "vector.hpp"
#include <initializer_list>

namespace dym {
template <typename Type, std::size_t m, std::size_t n> struct Matrix {
private:
  Vector<Type, n> a[m];

public:
  Matrix(const Type &num = 0) {
    for (auto &i : a)
      i = num;
  }
  Matrix(const std::initializer_list<std::vector<Type>> &v) {
    // Loop<int, m>(
    //     [&](auto i) { Loop<int, n>([&](auto j) { a[i][j] = v[i][j]; }); });
    // for (int i = 0; i < m; ++i)
    //   for (int j = 0; j < n; ++j) a[i][j] = v[i][j];
    short i = 0, j = 0;
    for (auto &obji : v) {
      for (auto &objij : obji)
        a[i][j++] = objij;
      ++i, j = 0;
    }
  }
  // template <typename... v_args>
  Matrix(const std::vector<std::vector<Type>> &v) {
    Loop<int, m>(
        [&](auto i) { Loop<int, n>([&](auto j) { a[i][j] = v[i][j]; }); });
    // for (int i = 0; i < m; ++i)
    //   for (int j = 0; j < n; ++j) a[i][j] = v[i][j];
  }
  Matrix(const std::vector<Vector<Type, m>> &v) {
    Loop<int, m>([&](auto i) { a[i] = v[i]; });
    // Loop<int, m>(
    //     [&](auto i) { Loop<int, n>([&](auto j) { a[i][j] = v[j][i]; }); });

    // for (int i = 0; i < m; ++i) a[i] = v[i];
  }
  Matrix(std::function<void(Type &)> fun) {
    Loop<int, m>([&](auto i) { Loop<int, n>([&](auto j) { fun(a[i][j]); }); });
    // for (int i = 0; i < m; ++i)
    //   for (int j = 0; j < n; ++j) fun(a[i][j]);
  }
  Matrix(std::function<void(Type &, int, int)> fun) {
    Loop<int, m>(
        [&](auto i) { Loop<int, n>([&](auto j) { fun(a[i][j], i, j); }); });
    // for (int i = 0; i < m; ++i)
    //   for (int j = 0; j < n; ++j) fun(a[i][j], i, j);
  }
  Matrix(std::function<void(Vector<Type, n> &)> fun) {
    Loop<int, m>([&](auto i) { fun(a[i]); });
    // for (auto &e : a) fun(e);
  }
  Matrix(std::function<void(Vector<Type, n> &, int)> fun) {
    Loop<int, m>([&](auto i) { fun(a[i], i); });
    // int i = 0;
    // for (auto &e : a) fun(e, i++);
  }
  template <std::size_t inRank_m, std::size_t inRank_n>
  Matrix(const Matrix<Type, inRank_m, inRank_n> &v, const Type &vul = 0) {
    constexpr int for_min = std::min(m, inRank_m);
    Loop<int, for_min>([&](auto i) { a[i] = v[i]; });
    // for (int i = 0; i < for_min; ++i) a[i] = v[i];
    for (int i = for_min; i < m && i < n; ++i)
      a[i][i] = vul;
  }

  Matrix(const Matrix<Type, m, n> &&v) { std::memcpy(a, v.a, sizeof(Matrix)); }
  Matrix(const Matrix<Type, m, n> &v) { std::memcpy(a, v.a, sizeof(Matrix)); }

  void show() const { std::cout << *this << std::endl; }
  _DYM_FORCE_INLINE_ Matrix &for_each(std::function<void(Type &)> func) {
    Loop<int, m>([&](auto i) { func(a[i]); });
    // for (auto &e : a) func(e);
    return *this;
  }
  _DYM_FORCE_INLINE_ Matrix &
  for_each(std::function<void(Type &, int, int)> func) {
    Loop<int, m>(
        [&](auto i) { a[i].for_each([&](Type &e, int j) { func(e, i, j); }); });
    // int i = 0;
    // for (auto &v : a) v.for_each([&](Type &e, int j) { func(e, i, j); }),
    // i++;
    return *this;
  }

  Vector<Type, n> &operator[](const int &i) { return a[i]; }
  Vector<Type, n> operator[](const int &i) const { return a[i]; }

  template <std::size_t inRank_m, std::size_t inRank_n>
  Matrix operator=(const Matrix<Type, inRank_m, inRank_n> &v) {
    constexpr int for_min = std::min(m, inRank_m);
    Loop<int, for_min>([&](auto i) { a[i] = v[i]; });
    // for (int i = 0; i < for_min; ++i) a[i] = v[i];
    return *this;
  }
  inline Matrix operator=(const Matrix &v) {
    memcpy(a, v.a, sizeof(Matrix));
    return *this;
  }

  Matrix operator=(const Type &num) {
    for (auto &v : a)
      v = num;
    return *this;
  }
  friend std::ostream &operator<<(std::ostream &output, const Matrix &v) {
    output << "Mat: [\n";
    for (auto &v : v.a) {
      for (int i = 0; i < n; ++i)
        output << v[i] << " ";
      output << "\n";
    }
    output << "]";
    return output;
  }
  template <typename cType>
  _DYM_FORCE_INLINE_ Matrix<cType, m, n> cast() const {
    Matrix<cType, m, n> o;
    Loop<int, m>(
        [&](auto i) { Loop<int, n>([&](auto j) { o[i][j] = a[i][j]; }); });
    // for (int i = 0; i < m; ++i)
    //   for (int j = 0; j < n; ++j) o[i][j] = a[i][j];
    return o;
  }

  constexpr auto shape() const { return gi((int)m, (int)n); }

  inline Matrix<Type, n, m> transpose() const {
    return Matrix<Type, n, m>([&](Type &e, int i, int j) { e = a[j][i]; });
  }
  inline Vector<Type, m> getColVec(const int &col) const {
    return Vector<Type, m>([&](Type &e, int i) { e = a[i][col]; });
  }
  inline void setColVec(const int &col, Vector<Type, m> v) {
    v.for_each([&](Type &e, int i) { a[i][col] = e; });
  }
  inline Matrix<Type, m - 1, n - 1> sub(const unsigned short &i,
                                        const unsigned short &j) const {
    return Matrix<Type, m - 1, n - 1>([&](Type &e, int ii, int jj) {
      e = a[ii >= i ? ii + 1 : ii][jj >= j ? jj + 1 : jj];
    });
  }

  _DYM_FORCE_INLINE_ Type det() const;

  inline Matrix<Type, m, n> inverse() const;

  _DYM_FORCE_INLINE_ Matrix<Type, m, n> inv() const { return inverse(); }

  _DYM_FORCE_INLINE_ Type trace() const {
    Type ans = 0;
    if constexpr (m < n)
      Loop<int, m>([&](auto i) { ans += a[i][i]; });
    else
      Loop<int, n>([&](auto i) { ans += a[i][i]; });
    return ans;
  }

  _DYM_FORCE_INLINE_ Type tr() const { return trace(); }
};

template <typename Type, std::size_t m, std::size_t n>
inline Vector<Type, m> operator*(const Matrix<Type, m, n> &ma,
                                 const Vector<Type, n> &ve) {
  return Vector<Type, m>([&](Type &e, int i) { e = vector::dot(ma[i], ve); });
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
  // Loop<int, m1>([&](auto r) {
  //   Loop<int, n1>([&](auto i) {
  //     for (int c = 0; c < n2; ++c) o[r][c] += a[r][i] * b[i][c];
  //   });
  // });
  for (int r = 0; r < m1; ++r)
    for (int i = 0; i < n1; ++i)
      for (int c = 0; c < n2; ++c)
        o[r][c] += a[r][i] * b[i][c];
  return o;
}

template <typename Type, std::size_t m, std::size_t n>
inline Matrix<Type, m, n> operator/(const Matrix<Type, m, n> &v1,
                                    const Matrix<Type, m, n> &v2) {
  return v1 * v2.inverse();
}

template <typename Type, std::size_t m, std::size_t n>
inline Matrix<Type, m, n> operator-(const Matrix<Type, m, n> &v) {
  return Matrix<Type, m, n>([&](Type &e, int i, int j) { e = -v[i][j]; });
}

#define _dym_matrix_type_operator_binary_(op)                                  \
  template <typename Type, std::size_t m, std::size_t n>                       \
  inline Matrix<Type, m, n> operator op(const Type &f,                         \
                                        const Matrix<Type, m, n> &s) {         \
    return Matrix<Type, m, n>(                                                 \
        [&](Type &e, int i, int j) { e = f op s[i][j]; });                     \
  }                                                                            \
  template <typename Type, std::size_t m, std::size_t n>                       \
  inline Matrix<Type, m, n> operator op(const Matrix<Type, m, n> &f,           \
                                        const Type &s) {                       \
    return Matrix<Type, m, n>(                                                 \
        [&](Type &e, int i, int j) { e = f[i][j] op s; });                     \
  }

_dym_matrix_type_operator_binary_(*);
_dym_matrix_type_operator_binary_(/);

#define _dym_matrix_operator_binary_(op)                                       \
  template <typename Type, std::size_t m, std::size_t n>                       \
  inline Matrix<Type, m, n> operator op(const Matrix<Type, m, n> &f,           \
                                        const Matrix<Type, m, n> &s) {         \
    return Matrix<Type, m, n>(                                                 \
        [&](Type &e, int i, int j) { e = f[i][j] op s[i][j]; });               \
  }                                                                            \
  _dym_matrix_type_operator_binary_(op);

#define _dym_matrix_operator_unary_(op)                                        \
  template <typename Type, std::size_t m, std::size_t n>                       \
  inline void operator op(Matrix<Type, m, n> &f,                               \
                          const Matrix<Type, m, n> &s) {                       \
    Loop<int, m>(                                                              \
        [&](auto i) { Loop<int, n>([&](auto j) { f[i][j] op s[i][j]; }); });   \
  }                                                                            \
  template <typename Type, std::size_t m, std::size_t n>                       \
  inline void operator op(Matrix<Type, m, n> &f, const Type &s) {              \
    Loop<int, m>(                                                              \
        [&](auto i) { Loop<int, n>([&](auto j) { f[i][j] op s; }); });         \
  }

_dym_matrix_operator_binary_(+);
_dym_matrix_operator_binary_(-);

_dym_matrix_operator_unary_(+=);
_dym_matrix_operator_unary_(-=);
_dym_matrix_operator_unary_(*=);
_dym_matrix_operator_unary_(/=);

} // namespace dym

/*
 * @Author: DyllanElliia
 * @Date: 2022-01-14 14:51:57
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-08 16:45:18
 * @Description:
 */
#pragma once
#include "matALG_others/matSVD.hpp"
#include "matrix.hpp"

namespace dym {
namespace matrix {
template <typename Type, std::size_t dim>
constexpr inline Matrix<Type, dim, dim> identity(Type vul = 1) {
  return Matrix<Type, dim, dim>(
      [&](Type &e, int i, int j) { e = i == j ? vul : 0; });
}

template <typename Type, std::size_t m1, std::size_t n1, std::size_t m2,
          std::size_t n2>
inline Matrix<Type, m1, n2> mul_std(Matrix<Type, m1, n1> &a,
                                    Matrix<Type, m2, n2> &b) {
  static_assert(n1 == m2,
                "Left Matrix's col must be equal to Right Matrix's row!");
  Matrix<Type, m1, n2> o(0);
  for (int r = 0; r < m1; ++r)
    for (int c = 0; c < n2; ++c)
      for (int i = 0; i < n1; ++i)
        o[r][c] += a[r][i] * b[i][c];
  return o;
}

template <typename Type, std::size_t m1, std::size_t n1, std::size_t m2,
          std::size_t n2>
inline Matrix<Type, m1, n2> mul_swap(Matrix<Type, m1, n1> &a,
                                     Matrix<Type, m2, n2> &b) {
  static_assert(n1 == m2,
                "Left Matrix's col must be equal to Right Matrix's row!");
  Matrix<Type, m1, n2> o(0);
  for (int r = 0; r < m1; ++r)
    for (int i = 0; i < n1; ++i)
      for (int c = 0; c < n2; ++c)
        o[r][c] += a[r][i] * b[i][c];
  return o;
}

template <typename Type, std::size_t m1, std::size_t n1, std::size_t m2,
          std::size_t n2>
inline Matrix<Type, m1, n2> mul_fast(Matrix<Type, m1, n1> &a,
                                     Matrix<Type, m2, n2> &b) {
  static_assert(n1 == m2,
                "Left Matrix's col must be equal to Right Matrix's row!");
  Matrix<Type, m1, n2> o(0);
  Type *dest = &(o[0][0]);
  const Type *srcA = &(a[0][0]), *srcB = &(b[0][0]);
  for (int i = 0; i < m1; ++i)
    for (int k = 0; k < n1; ++k) {
      const Type *na = srcA + i * n1 + k;
      const Type *nb = srcB + k * n2;
      Type *nc = dest + i * n2;

      Type *cMac = nc + n2;
      while (nc < cMac) {
        *nc++ += (*na) * (*nb++);
      }
    }
  return o;
}

template <typename Type, std::size_t dim>
constexpr inline Matrix<Type, dim, dim>
outer_product(const Vector<Type, dim> &a, const Vector<Type, dim> &b) {
  return Matrix<Type, dim, dim>(
      [&](Vector<Type, dim> &e, int i) { e = a[i] * b; });
}

template <typename Type, std::size_t dim>
inline Type det(const Matrix<Type, dim, dim> &mat) {
  if constexpr (dim == 1)
    return mat[0][0];
  if constexpr (dim == 2)
    return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
  else {
    Type ans = 0;
    if constexpr (dim <= 5)
      Loop<int, dim>([&](auto i) {
        ans += (i % 2 ? -1 : 1) * det(mat.sub(0, i)) * mat[0][i];
      });
    else
      for (int i = 0; i < dim; ++i)
        ans += (i % 2 ? -1 : 1) * det(mat.sub(0, i)) * mat[0][i];
    return ans;
  }
}

template <typename Type, std::size_t m, std::size_t n>
_DYM_FORCE_INLINE_ Matrix<Type, n, m>
transposed(const Matrix<Type, m, n> &mat) {
  return mat.transpose();
}

template <typename Type, std::size_t dim>
inline Matrix<Type, dim, dim> inversed(const Matrix<Type, dim, dim> &mat) {
  return mat.inverse();
}

template <typename Type, std::size_t m, std::size_t n>
_DYM_FORCE_INLINE_ Type tr(const Matrix<Type, m, n> &mat) {
  return mat.trace();
}

} // namespace matrix
namespace {
template <typename Type> Type getFirst(Type t) { return t; }
template <typename Type, typename... Vs> Type getFirst(Type t, Vs... vec) {
  return t;
}
} // namespace
template <typename Type, std::size_t dim>
template <typename... Vs>
inline Vector<Type, dim> Vector<Type, dim>::cross(Vs... vec) const {
  if constexpr ((std::is_same_v<Vs, Vector<Type, dim>> && ...) &&
                sizeof...(vec) == 1 && dim == 2) {
    auto obj = getFirst(vec...);
    return Vector<Type, dim>({0, a[0] * obj[1] - obj[0] * a[1]});
  }
  if constexpr ((std::is_same_v<Vs, Vector<Type, dim>> && ...) &&
                sizeof...(vec) == dim - 2) {
    if constexpr (dim == 1)
      return Vector<Type, dim>(Type(0));
    if constexpr (dim == 2)
      return Vector<Type, dim>({a[1], -a[0]});
    Matrix<Type, dim, dim> mat({Vector<Type, dim>((Type)0), *this, vec...});
    return Vector<Type, dim>([&](Type &v, int i) {
      v = (i % 2 ? -1 : 1) * matrix::det(mat.sub(0, i));
    });
  } else {
    qp_ctrl(tColor::RED, tType::BOLD, tType::UNDERLINE);
    qprint("Vector Error: please check the dimension of the input vector for "
           "the "
           "function cross.");
    qp_ctrl();
    return Vector(0);
  }
}

template <typename Type, std::size_t m, std::size_t n>
inline Matrix<Type, m, n> Matrix<Type, m, n>::inverse() const {
  static_assert(m == n, "Matrix Error: Only square matrix can be inverted!");
  auto &mat = *this;
  auto detMat = matrix::det(mat);
  if (dym::abs(detMat) < 1e-7)
    return Matrix<Type, m, n>(Type(0));
  auto detMat_inv = (Type)1 / detMat;
  return Matrix<Type, m, n>([&](Type &e, int i, int j) {
    e = ((i + j) % 2 ? -1 : 1) * detMat_inv * matrix::det(mat.sub(j, i));
  });
}

template <typename Type, std::size_t m, std::size_t n>
_DYM_FORCE_INLINE_ Type Matrix<Type, m, n>::det() const {
  static_assert(
      m == n, "Matrix Error: Only square matrix can compute the determinant!");
  return matrix::det(*this);
}
} // namespace dym
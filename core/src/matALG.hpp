/*
 * @Author: DyllanElliia
 * @Date: 2022-01-14 14:51:57
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-24 16:23:41
 * @Description:
 */
#pragma once
#include "matrix.hpp"
#include "matALG_others/matSVD.hpp"
namespace dym {
namespace matrix {
template <typename Type, std::size_t dim>
Matrix<Type, dim, dim> identity(Type vul = 1) {
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
      for (int i = 0; i < n1; ++i) o[r][c] += a[r][i] * b[i][c];
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
      for (int c = 0; c < n2; ++c) o[r][c] += a[r][i] * b[i][c];
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
inline Matrix<Type, dim, dim> outer_product(const Vector<Type, dim> &a,
                                            const Vector<Type, dim> &b) {
  return Matrix<Type, dim, dim>(
      [&](Vector<Type, dim> &e, int i) { e = a[i] * b; });
}

template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Type det(const Matrix<Type, dim, dim> &mat) {
  if constexpr (dim == 1) return mat[0][0];
  if constexpr (dim == 2)
    return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
  else {
    Type ans = 0;
    Loop<int, dim>([&](auto i) {
      ans += (i % 2 ? -1 : 1) * det(mat.sub(0, i)) * mat[0][i];
    });
    return ans;
  }
}

}  // namespace matrix

template <typename Type, std::size_t dim>
template <typename... Vs>
_DYM_FORCE_INLINE_ Vector<Type, dim> Vector<Type, dim>::cross(Vs... vec) {
  if constexpr ((std::is_same_v<Vs, Vector<Type, dim>> && ...) &&
                sizeof...(vec) == dim - 2) {
    if constexpr (dim == 1) return Vector<Type, dim>(Type(0));
    if constexpr (dim == 2) return Vector<Type, dim>({a[1], -a[0]});
    Matrix<Type, dim, dim> mat({Vector<Type, dim>((Type)0), *this, vec...});
    return Vector<Type, dim>([&](Type &v, int i) {
      v = (i % 2 ? -1 : 1) * matrix::det(mat.sub(0, i));
    });
  } else {
    qp_ctrl(tColor::RED, tType::BOLD, tType::UNDERLINE);
    qprint(
        "Vector Error: please check the input vector dimensions of "
        "function "
        "cross.");
    qp_ctrl();
    return Vector(0);
  }
}
}  // namespace dym
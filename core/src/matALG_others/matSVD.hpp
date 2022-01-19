/*
 * @Author: DyllanElliia
 * @Date: 2022-01-14 15:53:11
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-19 16:56:07
 * @Description:
 */
#pragma once
#include "../matrix.hpp"

#include "truncatedSvd.hpp"
#include "fast3x3SVD.hpp"

namespace dym {
namespace matrix {
namespace {

template <typename Type, std::size_t m, std::size_t n>
void truncatedSvd_v(Matrix<Type, m, n>& A, Matrix<Type, m, m>& U,
                    Matrix<Type, m, m>& Sig, Matrix<Type, n, m>& V) {
  if constexpr (m <= n) {
    Matrix<Type, (m <= n ? m : n), m> Ut;
    Matrix<Type, (m <= n ? m : n), n> Vt;
    truncatedSvd::svd_truncated_v(m, n, &(A[0][0]), &(Ut[0][0]), &(Sig[0][0]),
                                  &(Vt[0][0]));
    U = Ut.transpose(), V = Vt.transpose();
  } else
    printf("Matrix must m <= n");
}
template <typename Type, std::size_t m, std::size_t n>
void truncatedSvd_u(Matrix<Type, m, n>& A, Matrix<Type, m, n>& U,
                    Matrix<Type, n, n>& Sig, Matrix<Type, n, n>& V) {
  if constexpr (n <= m) {
    Matrix<Type, (m <= n ? m : n), m> Ut;
    Matrix<Type, (m <= n ? m : n), n> Vt;
    truncatedSvd::svd_truncated_u(m, n, &(A[0][0]), &(U[0][0]), &(Sig[0][0]),
                                  &(V[0][0]));
    U = Ut.transpose(), V = Vt.transpose();
  } else
    printf("Matrix must n <= m");
}
}  // namespace
template <typename Type, std::size_t m, std::size_t n>
void truncatedSvd(Matrix<Type, m, n>& A, Matrix<Type, m, (m <= n ? m : n)>& U,
                  Matrix<Type, (m <= n ? m : n), (m <= n ? m : n)>& Sig,
                  Matrix<Type, n, (m <= n ? m : n)>& V) {
  if constexpr (std::is_same<Type, Real>::value) {
    if constexpr (m <= n)
      truncatedSvd_v(A, U, Sig, V);
    else
      truncatedSvd_u(A, U, Sig, V);
  }
}

template <typename Type>
void fast3x3Svd(Matrix<Type, 3, 3>& A, Matrix<Type, 3, 3>& U,
                Matrix<Type, 3, 3>& Sig, Matrix<Type, 3, 3>& V) {
  if constexpr (std::is_same<Type, Real>::value) {
    fast3x3SVD::svd(
        A[0][0], A[0][1], A[0][2], A[1][0], A[1][1], A[1][2], A[2][0], A[2][1],
        A[2][2], U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0],
        U[2][1], U[2][2], Sig[0][0], Sig[0][1], Sig[0][2], Sig[1][0], Sig[1][1],
        Sig[1][2], Sig[2][0], Sig[2][1], Sig[2][2], V[0][0], V[0][1], V[0][2],
        V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);
  }
}

}  // namespace matrix
}  // namespace dym
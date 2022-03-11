/*
 * @Author: DyllanElliia
 * @Date: 2022-01-14 15:53:11
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-11 16:28:56
 * @Description:
 */
#pragma once
#include "../matrix.hpp"
#include "fast3x3SVD.hpp"
#include "traditionalSVD.hpp"
#include "truncatedSvd.hpp"

namespace dym {
namespace matrix {
namespace {

template <typename Type, std::size_t m, std::size_t n>
inline void truncatedSvd_v(Matrix<Type, m, n>& A, Matrix<Type, m, m>& U,
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
inline void truncatedSvd_u(Matrix<Type, m, n>& A, Matrix<Type, m, n>& U,
                           Matrix<Type, n, n>& Sig, Matrix<Type, n, n>& V) {
  if constexpr (n <= m) {
    Matrix<Type, (m <= n ? m : n), m> Ut;
    Matrix<Type, (m <= n ? m : n), n> Vt;
    Matrix<Type, n, n> St;
    truncatedSvd::svd_truncated_u(m, n, &(A[0][0]), &(U[0][0]), &(St[0][0]),
                                  &(V[0][0]));
    U = Ut.transpose(), V = Vt.transpose(), Sig = St.transpose();
  } else
    printf("Matrix must n <= m");
}
}  // namespace
template <typename Type, std::size_t m, std::size_t n>
inline void truncatedSvd(Matrix<Type, m, n>& A,
                         Matrix<Type, m, (m <= n ? m : n)>& U,
                         Matrix<Type, (m <= n ? m : n), (m <= n ? m : n)>& Sig,
                         Matrix<Type, n, (m <= n ? m : n)>& V) {
  // if constexpr (std::is_same<Type, Real>::value) {
  if constexpr (m <= n)
    truncatedSvd_v(A, U, Sig, V);
  else
    truncatedSvd_u(A, U, Sig, V);
  // }
}

template <typename Type>
inline void fast3x3Svd(Matrix<Type, 3, 3>& A, Matrix<Type, 3, 3>& U,
                       Matrix<Type, 3, 3>& Sig, Matrix<Type, 3, 3>& V) {
  if constexpr (std::is_same<Type, float>::value) {
    fast3x3SVD::svd(
        A[0][0], A[0][1], A[0][2], A[1][0], A[1][1], A[1][2], A[2][0], A[2][1],
        A[2][2], U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0],
        U[2][1], U[2][2], Sig[0][0], Sig[0][1], Sig[0][2], Sig[1][0], Sig[1][1],
        Sig[1][2], Sig[2][0], Sig[2][1], Sig[2][2], V[0][0], V[0][1], V[0][2],
        V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);
  } else {
    DYM_ERROR("Matrix SVD error: fast3x3Svd only support Matrix<float,3,3>!");
    exit(EXIT_FAILURE);
  }
}

template <typename Type>
inline void traditionalSvd(Matrix<Type, 3, 3>& A, Matrix<Type, 3, 3>& U,
                           Matrix<Type, 3, 3>& Sig, Matrix<Type, 3, 3>& V) {
  // if constexpr (std::is_same<Type, Real>::value)
  traditionalSVD ::svd(A, U, Sig, V);
}

template <typename Type, std::size_t m, std::size_t n>
inline void svd(Matrix<Type, m, n>& A, Matrix<Type, m, n>& U,
                Matrix<Type, m, n>& Sig, Matrix<Type, m, n>& V,
                bool use_fast3x3Svd = false) {
  if constexpr (m == 3 && n == 3) {
    if (use_fast3x3Svd)
      fast3x3Svd(A, U, Sig, V);
    else
      traditionalSvd(A, U, Sig, V);
  } else
    truncatedSvd(A, U, Sig, V);
}

template <typename Type, std::size_t n>
inline void pd(Matrix<Type, n, n>& A, Matrix<Type, n, n>& U,
               Matrix<Type, n, n>& P) {
  Matrix<Type, n, n> V;
  auto& Sig = P;
  svd(A, U, Sig, V);
  auto Vt = V.transpose();
  P = V * Sig * Vt;
  U = U * Vt;
}

}  // namespace matrix
}  // namespace dym
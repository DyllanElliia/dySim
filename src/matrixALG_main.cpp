/*
 * @Author: DyllanElliia
 * @Date: 2022-01-19 15:52:08
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-10 17:35:32
 * @Description:
 */

#include "../core/math/matALG.hpp"
#include <random>
#include "../core/tools/sugar.hpp"

int main(int argc, char** argv) {
  std::default_random_engine re;
  std::uniform_real_distribution<Real> u(0.f, 1.f);
  const int m = 3, n = 3;
  // dym::Matrix<Real, m, n> A([&](Real& e) { e = u(re); });
  dym::Matrix<Real, m, n> A({{3, 1, 0}, {1, 2, 2}, {0, 1, 1}});

  dym::Matrix<Real, m, (m <= n ? m : n)> U;
  dym::Matrix<Real, (m <= n ? m : n), (m <= n ? m : n)> Sig;
  dym::Matrix<Real, n, (m <= n ? m : n)> V;

  qprint("Transpose:");
  qprint(A, A.transpose());

  qprint("truncatedSvd:");
  dym::matrix::truncatedSvd(A, U, Sig, V);
  qprint(A, "U x SIg x V'=", U * Sig * V.transpose());
  qprint(U, Sig, V);

  qprint("fast3x3Svd:");
  dym::matrix::fast3x3Svd(A, U, Sig, V);
  qprint(A, "U x SIg x V'=", U * Sig * V.transpose());
  qprint(U, Sig, V);

  U = 0, Sig = 0, V = 0;
  qprint("traditionSvd:");
  dym::matrix::traditionalSvd(A, U, Sig, V);
  qprint(A, "U x SIg x V'=", U * Sig * V.transpose());
  qprint(U, Sig, V);

  U = 0, Sig = 0, V = 0;
  qprint("svd:");
  dym::matrix::svd(A, U, Sig, V);
  qprint(A, "U x SIg x V'=", U * Sig * V.transpose());
  qprint(U, Sig, V);

  dym::Matrix<Real, 3, 3> P;
  qprint("pd:");
  dym::matrix::pd(A, U, P);
  qprint(A, "U x P'=", U * P.transpose());
  qprint(U, P);
}
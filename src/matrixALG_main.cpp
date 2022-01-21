/*
 * @Author: DyllanElliia
 * @Date: 2022-01-19 15:52:08
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-21 15:09:57
 * @Description:
 */

#include "../core/src/matALG.hpp"
#include <random>
#include "../tools/sugar.hpp"

int main(int argc, char** argv) {
  std::default_random_engine re;
  std::uniform_real_distribution<float> u(0.f, 1.f);
  const int m = 3, n = 3;
  // dym::Matrix<float, m, n> A([&](float& e) { e = u(re); });
  dym::Matrix<float, m, n> A({{3, 1, 0}, {1, 2, 2}, {0, 1, 1}});

  dym::Matrix<float, m, (m <= n ? m : n)> U;
  dym::Matrix<float, (m <= n ? m : n), (m <= n ? m : n)> Sig;
  dym::Matrix<float, n, (m <= n ? m : n)> V;

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
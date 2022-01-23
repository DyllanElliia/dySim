/*
 * @Author: DyllanElliia
 * @Date: 2022-01-20 17:57:52
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-23 19:52:06
 * @Description:
 */

#include "../core/src/matALG.hpp"
#include <random>
#include "../tools/sugar.hpp"

Real errorCmp(dym::Matrix<Real, 3, 3>& a, dym::Matrix<Real, 3, 3>& b) {
  Real ans = 0.f;
  a.for_each([&](Real& e, int i, int j) { ans = std::abs(a[i][j] - b[i][j]); });
  return ans / 9;
}

int main(int argc, char const* argv[]) {
  std::default_random_engine re;
  std::uniform_real_distribution<float> u(-100.f, 100.f);

  int times = 100000;
  dym::TimeLog t;
  qprint("truncated Svd");
  t.reStart();
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](float&e) { e = u(re); }), U, Sig, V;
    dym::matrix::truncatedSvd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
  }
  t.record();

  qprint("\nfast 3x3 Svd");
  t.reStart();
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](float&e) { e = u(re); }), U, Sig, V;
    dym::matrix::fast3x3Svd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
  }
  t.record();

  qprint("\ntraditional 3x3 Svd");
  t.reStart();
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](float&e) { e = u(re); }), U, Sig, V;
    dym::matrix::traditionalSvd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
  }
  t.record();

  qprint("\ntruncated Svd");
  Real ans = 0.f;
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](float&e) { e = u(re); }), U, Sig, V;
    dym::matrix::truncatedSvd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
    ans += errorCmp(A, usv);
  }
  std::cout << "error: " << ans / times << "\n" << std::endl;

  ans = 0.f;
  qprint("fast 3x3 Svd");
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](float&e) { e = u(re); }), U, Sig, V;
    dym::matrix::fast3x3Svd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
    ans += errorCmp(A, usv);
  }
  std::cout << "error: " << ans / times << "\n" << std::endl;

  ans = 0.f;
  qprint("traditional 3x3 Svd");
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](float&e) { e = u(re); }), U, Sig, V;
    dym::matrix::traditionalSvd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
    ans += errorCmp(A, usv);
  }
  std::cout << "error: " << ans / times << "\n" << std::endl;

  return 0;
}

/*
 * @Author: DyllanElliia
 * @Date: 2022-01-20 17:57:52
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-29 16:30:44
 * @Description:
 */

#include <random>

#include "../core/math/matALG.hpp"
#include "../core/tools/sugar.hpp"

template <std::size_t m, std::size_t n>
Real errorCmp(dym::Matrix<Real, m, n>& a, dym::Matrix<Real, m, n>& b) {
  Real ans = 0.f;
  a.for_each([&](Real& e, int i, int j) { ans = std::abs(a[i][j] - b[i][j]); });
  return ans / (m * n);
}
template <std::size_t m, std::size_t n>
Real errorCmp(dym::Matrix<float, m, n>& a, dym::Matrix<float, m, n>& b) {
  Real ans = 0.f;
  a.for_each(
      [&](float& e, int i, int j) { ans = std::abs(a[i][j] - b[i][j]); });
  return ans / (m * n);
}

int main(int argc, char const* argv[]) {
  std::default_random_engine re;
  std::uniform_real_distribution<Real> u(-100.f, 100.f);

  int times = 100000;
  dym::TimeLog t;

  qprint("random");
  t.reStart();
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](Real&e) { e = u(re); }), U, Sig, V;
    // dym::matrix::truncatedSvd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
  }
  t.record();

  qprint("truncated Svd");
  t.reStart();
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](Real&e) { e = u(re); }), U, Sig, V;
    dym::matrix::truncatedSvd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
  }
  t.record();

  qprint("\nfast 3x3 Svd");
  t.reStart();
  for (int i = 0; i < times; ++i) {
    dym::Matrix<float, 3, 3> A([&](float&e) { e = u(re); }), U, Sig, V;
    dym::matrix::fast3x3Svd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
  }
  t.record();

  qprint("\ntraditional 3x3 Svd");
  t.reStart();
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](Real&e) { e = u(re); }), U, Sig, V;
    dym::matrix::traditionalSvd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
  }
  t.record();

  qprint("\ntruncated Svd");
  Real ans = 0.f;
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](Real&e) { e = u(re); }), U, Sig, V;
    dym::matrix::truncatedSvd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
    ans += errorCmp(A, usv);
  }
  std::cout << "error: " << ans / times << "\n" << std::endl;

  ans = 0.f;
  qprint("fast 3x3 Svd");
  for (int i = 0; i < times; ++i) {
    dym::Matrix<float, 3, 3> A([&](float&e) { e = u(re); }), U, Sig, V;
    dym::matrix::fast3x3Svd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
    ans += errorCmp(A, usv);
  }
  std::cout << "error: " << ans / times << "\n" << std::endl;

  ans = 0.f;
  qprint("traditional 3x3 Svd");
  for (int i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> A([&](Real&e) { e = u(re); }), U, Sig, V;
    dym::matrix::traditionalSvd(A, U, Sig, V);
    auto usv = U * Sig * V.transpose();
    ans += errorCmp(A, usv);
  }
  std::cout << "error: " << ans / times << "\n" << std::endl;

  return 0;
}

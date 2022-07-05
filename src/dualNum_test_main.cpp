/*
 * @Author: DyllanElliia
 * @Date: 2022-07-05 15:21:15
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-05 16:24:00
 * @Description:
 */
#include <dyMath.hpp>

template <typename type> type tryFun1(type x) {
  return 3.0 * dym::sqr(x) + 2.0 * x;
}

template <typename type> type tryFun2(type x) {
  return dym::sin(x) * dym::cos(x) - dym::tanh(x) +
         2 * x / dym::exp(x) * dym::sqrt(x) + dym::pow(x, 4) * dym::log10(x);
}

int main(int argc, char const *argv[]) {
  dym::DualNum<Real> asdf(0);
  Real x = 2.f;
  auto res = tryFun1(dym::DualNum<Real>{x, 1});
  qprint(res);

  qprint(tryFun2(dym::DualNum<Real>{0.5, 1}));

  dym::DualNum<dym::Vector3> point{{1.0, 2.0, 3.0}, 1};
  point.show();

  dym::Matrix3 mat{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto resmp = mat * point;
  qprint(resmp);
  qprint((mat * (dym::Vector3{1, 2, 3} + 1e3) - mat * dym::Vector3{1, 2, 3}) /
         1e3);
  return 0;
}
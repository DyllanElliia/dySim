/*
 * @Author: DyllanElliia
 * @Date: 2022-07-05 15:21:15
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-06 16:28:00
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

template <typename typex, typename typey, typename typez>
auto tryFun3(typex x, typey y, typez z) {
  return dym::sin(x) * dym::cos(y) - dym::tanh(z) +
         2 * x / dym::exp(y) * dym::sqrt(x * z) +
         dym::pow(x, 4) * dym::log10(y * z);
}

int main(int argc, char const *argv[]) {
  dym::Dual<Real> asdf(0);
  Real x = 2.f;
  auto res = tryFun1(dym::Dual<Real>{x, 1});
  qprint(res);

  qprint(tryFun2(dym::Dual<Real>{0.5, 1}));
  qprint();

  Real x3 = 0.3, y3 = 0.5, z3 = 0.7;
  qprint("1. ", tryFun3(dym::Dual<Real>{x3, 1}, dym::Dual<Real>{y3, 1},
                        dym::Dual<Real>{z3, 1}));
  qprint("dx ", tryFun3(dym::Dual<Real>{x3, 1}, y3, z3));
  qprint("dy ", tryFun3(x3, dym::Dual<Real>{y3, 1}, z3));
  qprint("dz ", tryFun3(x3, y3, dym::Dual<Real>{z3, 1}));
  qprint();
  dym::Dual<dym::Vector3> point{{1.0, 2.0, 3.0}, 1};
  point.show();

  dym::Matrix3 mat{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto resmp = mat * point;
  qprint(resmp);
  qprint((mat * (dym::Vector3{1, 2, 3} + 1e3) - mat * dym::Vector3{1, 2, 3}) /
         1e3);
  return 0;
}
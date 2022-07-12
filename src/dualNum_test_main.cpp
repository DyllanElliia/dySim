/*
 * @Author: DyllanElliia
 * @Date: 2022-07-05 15:21:15
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-12 16:45:44
 * @Description:
 */
#include <dyMath.hpp>
#include <tuple>

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

dym::Dual<Real> tryFun3d(dym::Dual<Real> x, dym::Dual<Real> y,
                         dym::Dual<Real> z) {
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
  qprint("dx. ", tryFun3(dym::Dual<Real>{x3, 1}, dym::Dual<Real>{y3, 0},
                         dym::Dual<Real>{z3, 0}));
  qprint("dy. ", tryFun3(dym::Dual<Real>{x3, 0}, dym::Dual<Real>{y3, 1},
                         dym::Dual<Real>{z3, 0}));
  qprint("dz. ", tryFun3(dym::Dual<Real>{x3, 0}, dym::Dual<Real>{y3, 0},
                         dym::Dual<Real>{z3, 1}));

  qprint("dx ", tryFun3(dym::Dual<Real>{x3, 1}, y3, z3));
  qprint("dy ", tryFun3(x3, dym::Dual<Real>{y3, 1}, z3));
  qprint("dz ", tryFun3(x3, y3, dym::Dual<Real>{z3, 1}));
  qprint();
  dym::Dual<Real> tx1 = x3, tx2 = y3, tx3 = z3;
  qprint("dx ", dym::AD::dx(tryFun3d, tx1, dym::AD::all(tx1, y3, z3)));
  qprint("dy ", dym::AD::dx(tryFun3d, tx2, dym::AD::all(tx1, tx2, tx3)));
  qprint("dz ", dym::AD::dx(tryFun3d, tx3, dym::AD::all(tx1, tx2, tx3)));

  dym::Dual<dym::Vector3> point{{1.0, 2.0, 3.0}, 1};
  point.show();

  dym::Matrix3 mat{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto resmp = mat * point;
  qprint(resmp);
  qprint((mat * (dym::Vector3{1, 2, 3} + 1e3) - mat * dym::Vector3{1, 2, 3}) /
         1e3);

  // int t1 = 114514;
  // char t2 = 'a';
  // dym::Dual<Real> t3 = 3;

  // std::tuple<int &, char &, dym::Dual<Real> &> a{t1, t2, t3};
  // dym::Loop<int, 3>([&](auto i) { qprint(std::get<i>(a)); });
  // dym::Loop<int, 3>([&](auto i) {
  //   dym::Loop<int, 3>([&](auto j) {
  //     qprint(std::get<i>(a), std::get<j>(a),
  //            (void *)&std::get<i>(a) == (void *)&std::get<j>(a));
  //   });
  // });
  return 0;
}
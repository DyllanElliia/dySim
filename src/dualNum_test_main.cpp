/*
 * @Author: DyllanElliia
 * @Date: 2022-07-05 15:21:15
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-19 16:24:59
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

auto tryFun3d(dym::Dual<Real> x, dym::Dual<Real> y, dym::Dual<Real> z,
              dym::Dual<Real> t) {
  return dym::sin(x) * dym::cos(y) - dym::tanh(z) +
         2 * x / dym::exp(y) * dym::sqrt(x * z) +
         dym::pow(x, 4) * dym::log10(y * z);
}

auto tryFun4d(dym::Vector<dym::Dual<Real>, 3> x) {
  auto &x1 = x[0], &x2 = x[1], &x3 = x[2];
  return dym::sqr(x1) + x1 * x2 + x2 * x3;
}

Real lam = 1.234, mu = 2.345;
auto tryFun5d(dym::Matrix<dym::Dual<Real>, 2, 2> G) {
  auto &euu = G[0][0], euv = G[0][1] + G[1][0], &evv = G[1][1];
  euv.A /= 2.0;
  return lam / 2.0 * dym::sqr(euu + evv) +
         mu * (dym::sqr(euu) + dym::sqr(euv) + dym::sqr(evv));
}

auto tryFun6d(dym::Vector<dym::Dual<Real>, 3> x,
              dym::Matrix<dym::Dual<Real>, 3, 3> A,
              dym::Vector<dym::Dual<Real>, 3> y) {
  return x.transpose() * A * y;
}

int main(int argc, char const *argv[]) {
  Real x = 2.f;
  // 3*x^2+2x
  qprint(tryFun1(dym::Dual<Real>{x, 1}));
  // sin(x)*cos(y)-tanh(x)+2*x/exp(x)*sqrt(x)+x^4*log_10(x)
  qprint(tryFun2(dym::Dual<Real>{0.5, 1}));
  qprint();
  // sin(x)*cos(y)-tanh(z)+2*x/exp(y)*sqrt(x*z)+x^4*log_10(y*z)
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
  // sin(x)*cos(y)-tanh(z)+2*x/exp(y)*sqrt(x*z)+x^4*log_10(y*z)
  dym::Dual<Real> tx1 = x3, tx2 = y3, tx3 = z3, tx4 = 1;
  qprint("dx ", dym::AD::dx(tryFun3d, tx1, dym::AD::all(tx1, y3, z3, tx4)));
  qprint("dy ", dym::AD::dx(tryFun3d, tx2, dym::AD::all(tx1, tx2, tx3, tx4)));
  qprint("dz ", dym::AD::dx(tryFun3d, tx3, dym::AD::all(tx1, tx2, tx3, tx4)));
  qprint("dt ", dym::AD::dx(tryFun3d, tx4, dym::AD::all(tx1, tx2, tx3, tx4)));
  auto [dx, dy, dz] = dym::AD::d(tryFun3d, dym::AD::fc(tx1, tx2, tx3),
                                 dym::AD::all(tx1, tx2, tx3, tx4));
  qprint(dx, dy, dz);
  qprint();
  // x1^2+x1*x2+x2*x3
  dym::Vector<dym::Dual<Real>, 3> t4x = {1, 2, 3};
  qprint("dV ", dym::AD::dx(tryFun4d, t4x, dym::AD::all(t4x)));
  qprint();
  // StVK: W(G)  = lambda/2*(e_uu+e_vv)^2+mu*(e_uu^2+e_uv^2+e_vv^2)
  //       dW/dG = 2*mu*G+lambda*trace(G)*I
  dym::Matrix<dym::Dual<Real>, 2, 2> G{{2, 3}, {3, 4}};
  qprint("dG ", dym::AD::dx(tryFun5d, G, dym::AD::all(G)));
  qprint("dG ", dym::Dual<Real>{2 * mu, 0} * G +
                    lam * G.trace() *
                        dym::matrix::identity<dym::Dual<Real>, 2>({1, 0}));
  qprint();
  dym::Dual<dym::Vector3> point{{1.0, 2.0, 3.0}, 1};
  dym::Matrix3 mat{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  qprint(mat * point);
  qprint((mat * (dym::Vector3{1, 2, 3} + 1e3) - mat * dym::Vector3{1, 2, 3}) /
         1e3);

  dym::Vector<dym::Dual<Real>, 3> x6{1, 2, 3};
  dym::Matrix<dym::Dual<Real>, 3, 3> A6(
      [&](dym::Dual<Real> &obj, int i, int j) { obj = i * 3 + j; });
  dym::Vector<dym::Dual<Real>, 3> y6{4, 5, 6};
  qprint(x6.transpose());
  dym::Vector<dym::Vector<dym::Dual<Real>, 3>, 3> testy;

  qprint("dx ", dym::AD::dx(tryFun6d, x6, dym::AD::all(x6, A6, y6)));
  qprint("dA ", dym::AD::dx(tryFun6d, A6, dym::AD::all(x6, A6, y6)));
  qprint("dy ", dym::AD::dx(tryFun6d, y6, dym::AD::all(x6, A6, y6)));
  auto [dxv, dAm, dyv] =
      dym::AD::d(tryFun6d, dym::AD::fc(x6, A6, y6), dym::AD::all(x6, A6, y6));
  qprint(dxv, dAm, dyv);

  return 0;
}
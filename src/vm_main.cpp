/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 15:30:45
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-12 16:51:22
 * @Description:
 */
#include "../core/src/matrix.hpp"
#include <dyMath.hpp>

int main(int argc, char** argv) {
  dym::Vector<float, 3> a(1);
  dym::Vector<float, 3> b({1, 2, 3});
  std::cout << a + b << a * b << a - b << a / b << b + 1.f << b - 1.f << b * 2.f
            << b / 2.f;
  auto c = b;
  c.show();
  c *= 10.f;
  c.show();
  auto d = dym::Vector<float, 5>(b, 6);
  d.show();
  auto e = dym::Vector<float, 2>(d);
  e.show();
  dym::Vector<float, 10> z(10);
  a = z;
  a.show();
  a = e;
  a.show();
  a.cast<int>().show();
  dym::Vector<float, 10> g([](float& e, int i) { e = i; });
  g.show();
  g = 0;
  g.show();

  qprint("matrix test:");
  dym::Matrix<float, 3, 4> ma(1);
  auto mb = dym::Matrix<float, 2, 5>(ma, 10);
  auto mc = dym::Matrix<float, 4, 3>(ma, 10);
  auto mg = dym::Matrix<float, 8, 8>(ma, 10);

  auto me = mb;
  dym::Matrix<float, 2, 5> ml([](float& e, int i, int j) { e = i * 10 + j; });
  ma.show();
  mb.show();
  mc.show();
  mg.show();
  me.show();
  ml.show();
  ml.cast<int>().show();
  dym::Matrix<int, 10, 10>(ml.cast<int>(), -100).show();
  dym::Matrix<int, 2, 3>({{1, 2, 3}, {4, 5, 6}}).show();
  (ml * d).show();
  (dym::Matrix<int, 3, 2>({{1, 3}, {4, 0}, {2, 1}}) *
   dym::Vector<int, 2>({1, 5}))
      .show();
  return 0;
}
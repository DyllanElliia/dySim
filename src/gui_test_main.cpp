/*
 * @Author: DyllanElliia
 * @Date: 2022-01-23 19:33:53
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-23 19:59:04
 * @Description:
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dyGraphic.hpp>

#include <random>
#include <string>

int main() {
  std::default_random_engine re;
  std::uniform_real_distribution<Real> u(0.f, 1.f);
  float t1 = 0, t2 = 0;
  int count = 0;
  dym::Tensor<Real> pos(0, dym::gi(10000, 2));
  pos.for_each_i([&](Real &e) { e = u(re); });

  dym::Tensor<Real> pos2(0, dym::gi(10000, 2));
  pos2.for_each_i([&](Real &e, int i, int j) {
    if (j == 1)
      e = -u(re);
    else
      e = u(re);
  });

  dym::GUI gui("dymathTest", dym::gi(0, 100, 100));
  gui.init(400, 400);
  dym::TimeLog tt;
  gui.update([&]() {
    auto pos1_ = pos * std::sin(t1), pos2_ = pos2 * std::sin(t2);
    t1 += 3e-2, t2 += 2e-2, ++count;
    gui.scatter2D(pos1_, dym::gi(255, 100, 0));
    gui.scatter2D(pos2_, dym::gi(0, 100, 255));
  });
  qprint("run counter: " + std::to_string(count));
  return 0;
}
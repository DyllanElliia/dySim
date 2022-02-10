/*
 * @Author: DyllanElliia
 * @Date: 2022-01-23 19:33:53
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-09 15:10:57
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
  dym::Tensor<dym::Vector<Real, 2>> pos(0, dym::gi(10000));
  pos.for_each_i([&](dym::Vector<Real, 2> &e) { e[0] = u(re), e[1] = u(re); });
  dym::Tensor<dym::Vector<Real, 2>> pos2(0, dym::gi(10000));
  pos2.for_each_i(
      [&](dym::Vector<Real, 2> &e, int i) { e[0] = u(re), e[1] = -u(re); });

  // GUI part:
  dym::GUI gui("dymathTest", dym::gi(0, 100, 100));
  gui.init(400, 400);
  dym::TimeLog tt;  // timer
  gui.update([&]() {
    auto pos1_ = pos * std::sin(t1), pos2_ = pos2 * std::sin(t2);
    t1 += 3e-2, t2 += 2e-2, ++count;
    gui.scatter2D(pos1_, dym::gi(255, 100, 0));
    gui.scatter2D(pos2_, dym::gi(0, 100, 255));
  });
  qprint("run counter: " + std::to_string(count));
  return 0;
}
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dyGraphic.hpp>

#include <random>
#include <string>

int main() {
  std::default_random_engine re;
  std::uniform_real_distribution<float> u(0.f, 1.f);
  float t1 = 0, t2 = 0, count = 0;
  dym::Tensor<float> pos(0, dym::gi(10000, 2));
  pos.for_each_i([&](float &e) { e = u(re); });

  dym::Tensor<float> pos2(0, dym::gi(10000, 2));
  pos2.for_each_i([&](float &e, int i, int j) {
    if (j == 1)
      e = -u(re);
    else
      e = u(re);
  });

  dym::GUI gui("dymathTest", dym::gi(0, 100, 100));
  gui.init(400, 400);
  dym::TimeLog tt;
  gui.update([&]() {
    // getchar();
    qprint("in");
    auto pos_ = pos * std::sin(t1);
    auto pos2_ = pos2 * std::sin(t2);
    t1 += 3e-2;
    t2 += 2e-2;
    // pos.show();
    ++count;
    gui.scatter2D(pos_, dym::gi(255, 100, 0));
    gui.scatter2D(pos2_, dym::gi(0, 100, 255));
  });
  qprint("run counter: " + std::to_string(count));
  return 0;
}
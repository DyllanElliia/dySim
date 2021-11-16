#include "src/tensor.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dyGraphic.hpp>

#include <random>
#include <string>

int main() {
  std::default_random_engine re;
  std::uniform_real_distribution<float> u(0.f, 1.f);
  float t = 0, count = 0;
  dym::Tensor<float> pos(0, dym::gi(10000, 2));
  pos.for_each_i([&](float &e) { e = u(re); });

  dym::GUI gui("dymathTest", dym::gi(0, 100, 100));
  gui.init(400, 400);
  dym::TimeLog tt;
  gui.update([&]() {
    // getchar();
    qprint("in");
    auto pos_ = pos * std::sin(t);
    t += 1e-2;
    // pos.show();
    ++count;
    gui.scatter2D(pos_, dym::gi(255, 100, 0), 0, 50);
    gui.scatter2D(pos_, dym::gi(0, 100, 255), 50);
  });
  qprint("run counter: " + std::to_string(count));
}
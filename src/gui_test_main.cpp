#include "src/tensor.hpp"
#include <cstdio>
#include <cstdlib>
#include <dyGraphic.hpp>

int main() {
  qprint("begin");
  dym::GUI gui("dymathTest", dym::gi(0, 100, 100));
  gui.init(400, 300);
  qprint("?");
  gui.update([&]() {
    getchar();
    qprint("in");
    dym::Tensor<float> pos({{10, 200}, {50, 150}, {100, 50}, {200, 200}});
    pos /= 210;
    pos.show();
    gui.scatter2D(pos, dym::gi(255, 100, 0));
  });
}
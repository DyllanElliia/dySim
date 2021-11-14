#include "src/tensor.hpp"
#include <dyGraphic.hpp>

int main() {
  qprint("begin");
  dym::GUI gui;
  gui.init(400, 300);
  qprint("?");
  gui.update([&]() {
    qprint("in");
    dym::Tensor<float> pos({{10, 200}, {50, 150}, {100, 50}, {200, 200}});
    pos.show();
    gui.scatter2D(pos, dym::gi(255, 100, 0));
  });
}
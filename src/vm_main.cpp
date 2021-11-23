/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 15:30:45
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-11-23 16:26:04
 * @Description:
 */
#include "../core/src/vector.hpp"

int main(int argc, char** argv) {
  dym::Vector<float, 3> a(1);
  dym::Vector<float, 3> b(1, 2, 3);
  std::cout << a + b << a * b << a - b << a / b << b + 1.f << b - 1.f << b * 2.f
            << b / 2.f;
  auto c = b;
  c.show();
  c *= 10.f;
  c.show();
  auto d = dym::Vector<float, 4>(b, 6);
  d.show();
  auto e = dym::Vector<float, 2>(d);
  e.show();
  return 0;
}
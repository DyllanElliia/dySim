/*
 * @Author: DyllanElliia
 * @Date: 2021-09-22 15:36:00
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-08 16:58:05
 * @Description: How to use Matrix.
 */

#include "../dyMath.hpp"
#include <iostream>

int main() {
  qprint("\033[1;4;33mMatrix example:", "Author: DyllanElliia",
         "Description: How to use Matrix.\033[0m");

  // Matrix is the subClass of Tensor.
  dym::Matrix<float> ma(dym::gi(4, 3), []() -> std::vector<float> {
    std::vector<float> v(4 * 3);
    int count = 0;
    for (auto &i : v)
      i = count++;
    return v;
  });

  // You can use vector<vector> to create it.
  // Forgive me, I have no idea how to use [][] initialize Matrix directly.
  // Maybe, it would be supported in the future.
  std::vector<std::vector<float>> v{{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}};
  dym::Matrix<float> mb(v);

  ma[dym::gi(1, 1)] = 0;
  qprint(ma, mb);

  // Show you how to use Matrix here.
  qprint("operator test!");
  dym::Matrix<float> mc = ma * mb * dym::Matrix<float>(dym::gi(4, 4), 1);
  // You can know the shape.
  qprint(dym::pi(mc.shape()), mc);

  // mc = ma * mb * Matrix<float>(gi(4, 4), 1);
  mc = mc - 1;
  qprint(mc);
  (2 * mc).show();
  (mc / 100).show();
  (mc + mc + mc - mc / 2).show();
  ((mc + mc + mc - mc / 2) * mc).show();

  mc = dym::Matrix<float>(v);
  qprint(mc);

  // You can use [int] and [Index] to access the Matrix.
  qprint(mc[2]);

  std::for_each(mc.begin(), mc.end(), [](float &i) { qprint_nlb(i); });
  qprint();

  for (auto i = mc.begin(); i != mc.end(); ++i)
    qprint_nlb(*i);
  qprint();

  return 0;
}
/*
 * @Author: DyllanElliia
 * @Date: 2021-09-22 15:36:00
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-09-24 16:13:04
 * @Description:
 */

#include "../dyMath.hpp"

int main() {
  // Matrix is the subClass of Tensor.
  Matrix<float> ma(gi(4, 3), []() -> std::vector<float> {
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
  Matrix<float> mb(v);

  ma[gi(1, 1)] = 0;
  ma.show();
  mb.show();

  // Show you how to use Matrix here.
  std::cout << "operator test!" << std::endl;
  Matrix<float> mc = ma * mb * Matrix<float>(gi(4, 4), 1);
  // You can know the shape.
  std::cout << pi(mc.shape()) << std::endl;
  mc.show();
  // mc = ma * mb * Matrix<float>(gi(4, 4), 1);
  mc = mc - 1;
  mc.show();
  (2 * mc).show();
  (mc / 100).show();
  (mc + mc + mc - mc / 2).show();
  ((mc + mc + mc - mc / 2) * mc).show();

  mc = Matrix<float>(v);
  mc.show();
  return 0;
}
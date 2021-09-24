/*
 * @Author: DyllanElliia
 * @Date: 2021-09-22 14:21:25
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-09-22 15:48:49
 * @Description: How to use Tensor
 */

#include "../dyMath.hpp"

int main() {
  // First parameter is the shape of the Tensor.
  // Second parameter is the default value.
  Tensor<int> x(gi(2, 2, 2), 1);

  // You can use show to print the Tensor.
  x.show();

  // Use Lambda function to create a Tensor.
  Tensor<int> a(gi(3, 2, 3), []() -> std::vector<int> {
    int x = 3, y = 4;
    std::vector<int> v(3 * 2 * 3, 2);
    v[x] = y;
    return v;
  });

  // You can use gi(index...) and Tensor[] to access the value.
  a[gi(1, 0, 0)] = 5;
  a[gi(2, 1, 1)] = 3;

  // You can use std::cout to print the Tensor.
  std::cout << a;

  // Use Lambda function to create a Tensor with shape.
  Tensor<int> b(gi(3, 2), [](const Index &shape) -> std::vector<int> {
    std::vector<int> v(3 * 2 * 3, 2);
    v[i2ti(gi(2, 1), shape)] = 9;
    return v;
  });
  b.show();

  // dyMath provides operator + - Tensor*Value Tensor/Value.
  // Operator Tensor*Tensor would be supported in the future.
  // Plz avoid confusing expression e.g. 2*a/2*2+1+a-1+2
  b = ((2 * a) / 2) * 2 + (2 - a);
  b.show();

  // Index provides operator + - * /.
  // pi() can print the Index's values.
  std::cout << pi(gi(3, 2, 1) * gi(2, 2, 2)) << std::endl;

  // You can use = to create a Tensor.
  Tensor<int> c = a / 2;
  c.show();

  // You can use Tensor.shape() to get the shape.
  std::cout << pi(c.shape()) << std::endl;

  // You can use cut(Index_begin, Index_end) to cut the Tensor.
  c = c.cut(gi(1, 0, 1), gi(2, 2, 3));
  std::cout << c << std::endl;

  return 0;
}
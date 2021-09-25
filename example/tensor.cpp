/*
 * @Author: DyllanElliia
 * @Date: 2021-09-22 14:21:25
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-09-25 23:21:22
 * @Description: How to use Tensor.
 */

#include "../dyMath.hpp"
#include <algorithm>
#include <iostream>
#include <ostream>

int main() {
  qprint("\033[1;4;33mTensor example:", "Author: DyllanElliia",
         "Description: How to use Tensor.\033[0m");

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

  // You can use gi(index...) and Tensor[Index] to access the value.
  a[gi(1, 0, 0)] = 5;
  a[gi(2, 1, 1)] = 3;
  // [int] also can be used. And it is faster than [Index].
  a[0] = 10;

  // You can use std::cout to print the Tensor.
  std::cout << a << std::endl;

  // Use Lambda function to create a Tensor with shape.
  Tensor<int> b(gi(3, 2), [](const Index &shape) -> std::vector<int> {
    std::vector<int> v(3 * 2 * 3, 2);
    v[i2ti(gi(2, 1), shape)] = 9;
    return v;
  });

  // You can use qprint to print the Tensor.
  qprint(b);

  // dyMath provides operator + - Tensor*Value Tensor/Value.
  // Operator Tensor*Tensor would be supported in the future.
  // Plz avoid confusing expression e.g. 2*a/2*2+1+a-1+2
  b = ((2 * a) / 2) * 2 + (2 - a);
  qprint(b);

  // Index provides operator + - * /.
  // pi() can print the Index's values.
  qprint(pi(gi(3, 2, 1) * gi(2, 2, 2)));

  // You can use = to create a Tensor.
  Tensor<int> c = a / 2;
  qprint(c);

  // You can use Tensor.shape() to get the shape.
  qprint(pi(c.shape()));

  // You can use cut(Index_begin, Index_end) to cut the Tensor.
  c = c.cut(gi(1, 0, 1), gi(2, 2, 3));
  qprint(c);

  std::for_each(b.begin(), b.end(), [](int &i) { qprint_nlb(i); });
  qprint();

  for (auto i = b.begin(); i != b.end(); ++i)
    qprint_nlb(*i);
  qprint();

  return 0;
}
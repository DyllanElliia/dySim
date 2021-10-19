/*
 * @Author: DyllanElliia
 * @Date: 2021-09-22 14:21:25
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-19 17:18:02
 * @Description: How to use Tensor.
 */

#include "../dyMath.hpp"
#include <algorithm>
#include <iostream>
#include <ostream>

int main() {
  qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::YELLOW);
  qprint("Tensor example:");
  qprint("Author: DyllanElliia");
  qprint("Description: How to use Tensor.");
  qp_ctrl();
  dym::TimeLog T;

  qprint("1. Create a Tensor");
  qprint("1.1. Directly to create");
  // First parameter is the default value.
  // Second parameter is the shape of the Tensor.
  dym::Tensor<int> x(1, dym::gi(2, 2, 2));
  // You can use show to print the Tensor.
  x.show();

  qprint("1.2. Use lambda");
  // Use Lambda function to create a Tensor.
  dym::Tensor<int> a(dym::gi(3, 2, 3), []() -> std::vector<int> {
    int x = 3, y = 4;
    std::vector<int> v(3 * 2 * 3, 2);
    v[x] = y;
    return v;
  });

  // You can use gi(index...) and Tensor[Index] to access the value.
  a[dym::gi(1, 0, 0)] = 5;
  a[dym::gi(2, 1, 1)] = 3;
  // [int] also can be used. And it is faster than [Index].
  a[0] = 10;

  // You can use std::cout to print the Tensor.
  std::cout << a << std::endl;

  // Use Lambda function to create a Tensor with shape.
  dym::Tensor<int> b(dym::gi(3, 2),
                     [](const dym::Index &shape) -> std::vector<int> {
                       std::vector<int> v(3 * 2, 2);
                       // function i2ti can transform Index to int.
                       v[dym::i2ti(dym::gi(2, 1), shape)] = 9;
                       return v;
                     });

  // You can use qprint to print the Tensor.
  qprint(b);

  qprint("1.3. Use function cut");
  dym::Tensor<int> acut = a.cut(dym::gi(0, 0, 0), dym::gi(1, 2, 3));
  acut.show();

  qprint("2. Tensor transpose: ");
  qprint("2.1. Use function t() to transpose 2-dimensions tensor (matrix)");
  acut.t().show();

  qprint("2.2. Use function transpose() to transpose n-dimensions tensor");
  // You can use function transpose to transpose the Tensor.
  // Note: both input parameters must be continuous!
  qprint(a.transpose(1, 2));
  qprint(a.transpose(0, 1));
  // test it!
  auto at = a.transpose(0, 1);
  qprint(at[dym::gi(0, 1, 0)], at[dym::gi(1, 2, 1)], a[dym::gi(0, 1, 0)],
         at[dym::gi(1, 0, 0)]);

  qprint("3. tensor dot-product");
  qprint("3.1. 2x3 * mx2x3");
  dym::Tensor<int> a2 = a.cut(dym::gi(0, 0, 0), dym::gi(1, 2, 3));
  a2.show();
  qprint(a * a2);

  dym::Tensor<int> a3({{1, 2, 3}, {1, 2, 3}});
  a3.show();
  qprint(a * a3);

  qprint("3.2. 2x1 * mx2x3");
  dym::Tensor<int> a4({1, 2});
  (a4 = a4.t()).show();
  qprint(a * a4);

  qprint("3.3. 1x1 * mx2x3");
  dym::Tensor<int> a5(2);
  qprint(a * a5);

  qprint("4. mathematical notation for tensors");
  // dyMath provides operator + - Tensor*Value Tensor/Value.
  // Operator Tensor*Tensor would be supported in the future.
  // Plz avoid confusing expression e.g. 2*a/2*2+1+a-1+2
  b = ((2 * a) / 2) * 2 + (2 - a);
  qprint(b);

  // Index provides operator + - * /.
  // pi() can print the Index's values.
  qprint(dym::pi(dym::gi(3, 2, 1) * dym::gi(2, 2, 2)));

  // You can use = to create a Tensor.
  dym::Tensor<int> c = a / 2;
  qprint(c);

  // You can use Tensor.shape() to get the shape.
  qprint(dym::pi(c.shape()));

  // You can use cut(Index_begin, Index_end) to cut the Tensor.
  c = c.cut(dym::gi(1, 0, 1), dym::gi(2, 2, 3));
  qprint(c);

  qprint("5. for_each elements of tensor");
  qprint("5.1. std::for_each");
  std::for_each(b.begin(), b.end(), [](int &i) { qprint_nlb(i); });
  qprint();

  qprint("5.2. for(auto i:tensor)");
  for (auto i = b.begin(); i != b.end(); ++i)
    qprint_nlb(*i);
  qprint();

  qprint("5.3.1. (recommented) tensor.for_each(element&)");
  auto b_s = b;
  int ma = -100;
  b.for_each([&ma](int &i) {
    i = -i;
    ma = std::max(ma, i);
  });
  (b - ma).show();
  qprint(ma);

  qprint("5.3.2. (recommented) tensor.for_each(element&, int i)");
  b = b_s;
  b.for_each([&b](int &e, int i) { qprint(e, i, b[i]); });

  qprint("5.3.3. (recommented) 2-D tensor.for_each(element&, int i, int j)");
  a3.for_each(
      [&a3](int &e, int i, int j) { qprint(e, i, j, a3[dym::gi(i, j)]); });

  qprint("5.3.4. (recommented) 3-D tensor.for_each(element&, int i, int j, int "
         "k)");
  b.for_each([&b](int &e, int i, int j, int k) {
    qprint(e, i, j, k, b[dym::gi(i, j, k)]);
  });

  qprint("5.3.5. (not-recommented) n-D tensor.for_each(element&, Index i)");
  dym::Tensor<int> m5d(2, dym::gi(2, 3, 4, 3, 2));
  m5d[dym::gi(0, 0, 0, 0, 0)] = 0;
  m5d[dym::gi(0, 1, 1, 0, 0)] = 5;
  m5d[dym::gi(1, 1, 0, 1, 0)] = 10;
  m5d[dym::gi(1, 1, 2, 1, 1)] = 15;
  m5d[dym::gi(1, 2, 3, 2, 1)] = 20;
  m5d.for_each([&m5d](int &e, dym::Index i) { qprint(e, dym::pi(i), m5d[i]); });

  return 0;
}

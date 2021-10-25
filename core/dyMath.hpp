/*
 * @Author: DyllanElliia
 * @Date: 2021-10-06 17:00:50
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-19 20:35:23
 * @Description:
 */

#pragma once

#pragma GCC optimize(2)

// #include "./picture.hpp"
#include "src/tensor.hpp"

namespace dym {
template <class InputType> Tensor<InputType> abs(Tensor<InputType> in) {
  in.for_each([](InputType &e) {
    if (e < 0)
      e = -e;
  });
  return in;
}
}; // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2021-10-06 17:00:50
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-11-09 11:51:44
 * @Description:
 */

#pragma once

// #include "./picture.hpp"
#include "src/tensor.hpp"

namespace dym {
template <class InputType>
Tensor<InputType> abs(Tensor<InputType> in) {
  in.for_each_i([](InputType &e) {
    if (e < 0) e = -e;
  });
  return in;
}
};  // namespace dym
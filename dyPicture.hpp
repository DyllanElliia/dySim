/*
 * @Author: DyllanElliia
 * @Date: 2021-09-13 16:50:00
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-08 16:50:30
 * @Description:
 */
#pragma once

#include "./picture.hpp"
// #include "./matrix.hpp"

namespace dym {

enum BorderType { BORDER_CONSTANT = 1, BORDER_REFLECT, BORDER_REPLICATE };

template <class InputType, class KernelType, int color_size>
Picture<InputType, color_size>
filter2D(Picture<InputType, color_size> &in, KernelType &kernel,
         BorderType border = BORDER_CONSTANT,
         Pixel<InputType, color_size> DefaultColor =
             Pixel<InputType, color_size>(0)) {
  Index vShape = in.shape(), vBorder1 = vShape - gi(1, 1, 0);
  auto &vShapeX = vShape[0], &vShapeY = vShape[1];
  auto &vBorder1X = vBorder1[0], &vBorder1Y = vBorder1[1];
  std::map<BorderType,
           std::function<Pixel<InputType, color_size>(
               int, int, int &, int &, Picture<InputType, color_size> &)>>
      gvm = {
          {BORDER_CONSTANT,
           [](int x, int y, int &vBorder1X, int &vBorder1Y,
              Picture<InputType, color_size> &in) {
             if (x < 0 || x > vBorder1X || y < 0 || y > vBorder1Y)
               return in[gi(0, 0)];
             return in[gi(x, y)];
           }},
          {BORDER_REFLECT,
           [=](int x, int y, int &vBorder1X, int &vBorder1Y,
               Picture<InputType, color_size> &in) {
             return in[gi((x < 0 ? 0 : (x > vBorder1X ? vBorder1X : x)),
                          (y < 0 ? 0 : (y > vBorder1Y ? vBorder1Y : y)))];
           }},
          {BORDER_REPLICATE, [=](int x, int y, int &vBorder1X, int &vBorder1Y,
                                 Picture<InputType, color_size> &in) {
             while (x < 0 || x > vBorder1X)
               x = x < 0 ? -x : (x > vBorder1X ? 2 * vBorder1X - x : x);
             while (y < 0 || y > vBorder1Y)
               y = y < 0 ? -y : (y > vBorder1Y ? 2 * vBorder1Y - y : y);
             return in[gi(x, y)];
           }}};
  auto &gv = gvm[border];

  Picture<InputType, color_size> result(gi(vShapeX, vShapeY));

  Index kShape = kernel.shape();
  auto &kShapeX = kShape[0], &kShapeY = kShape[1];
  auto kCenterX = kShapeX >> 1, kCenterY = kShapeY >> 1;
  auto kSize = kShapeX * kShapeY;

  for (int i = 0; i < vShapeX; ++i)
    for (int j = 0; j < vShapeY; ++j) {
      auto &rij = result[gi(i, j)];
      rij = 0;
      for (int ki = 0; ki < kShapeX; ++ki)
        for (int kj = 0; kj < kShapeY; ++kj) {
          rij += gv(i - kCenterX + ki, j - kCenterY + kj, vBorder1X, vBorder1Y,
                    in) *
                 kernel[gi(ki, kj)];
          // qprint_nlb(in[gi(i, j)],
          //            gv(i - kCenterX + ki, j - kCenterY + kj, vBorder1X,
          //               vBorder1Y, in),
          //            i, j, i - kCenterX + ki, j - kCenterY + kj, ki, kj,
          //            kernel[gi(ki, kj)]);
          // qprint();
        }
      // qprint(result[gi(i, j)]);
      // return result;
    }

  return result;
}

template <class InputType, int color_size>
Picture<InputType, color_size> abs(Picture<InputType, color_size> in) {
  Index vShape = in.shape();
  auto &vShapeX = vShape[0], &vShapeY = vShape[1];
  Picture<InputType, color_size> result(in);
  result.for_each([](Pixel<InputType, color_size> &i) {
    for (int j = 0; j < color_size; ++j)
      if (i[j] < 0)
        i[j] = -i[j];
  });

  return result;
}

} // namespace dym

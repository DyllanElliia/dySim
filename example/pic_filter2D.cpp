/*
 * @Author: DyllanElliia
 * @Date: 2021-09-30 17:00:48
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-09-30 17:04:28
 * @Description:
 */

#include "../dyMath.hpp"

int main() {
  // Read Picture
  Picture<float, 1> pic;
  pic.imread("./image/luna.png");

  Picture<float, 1> pic_L, pic_S;

  // Create Laplacian kernel
  Matrix<float> kernel_L(gi(3, 3), -1);
  kernel_L[gi(1, 1)] = 8;
  qprint(kernel_L);
  // run!
  pic_L = dym::filter2D(pic, kernel_L, dym::BORDER_REPLICATE);

  // Create Roberts kernel
  Matrix<float> kernel_S1(gi(2, 2), []() {
    std::vector<float> v{-1, 0, 0, 1};
    return v;
  });
  qprint(kernel_S1);
  Matrix<float> kernel_S2(gi(2, 2), []() {
    std::vector<float> v{0, -1, 1, 0};
    return v;
  });
  qprint(kernel_S2);
  // run!
  pic_S = dym::abs(dym::filter2D(pic, kernel_S1, dym::BORDER_REPLICATE)) +
          dym::abs(dym::filter2D(pic, kernel_S2, dym::BORDER_REPLICATE));

  // Save Picture
  pic.imwrite("./image_out/p1.png");
  pic_L.imwrite("./image_out/p_L.png");
  pic_S.imwrite("./image_out/p_S.png");
}
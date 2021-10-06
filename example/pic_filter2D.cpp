/*
 * @Author: DyllanElliia
 * @Date: 2021-09-30 17:00:48
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-06 17:02:58
 * @Description:
 */

#include "../dyPicture.hpp"

int main() {
  qprint("\033[1;4;33mPicture_filter2D example:", "Author: DyllanElliia",
         "Description: How to use filter2D.\033[0m");

  // Read Picture
  Picture<float, 1> pic;
  pic.imread("./image/luna.png");

  Picture<float, 1> pic_L, pic_S;

  // Create Laplacian kernel
  Matrix<float> kernel_L(gi(3, 3), -1);
  kernel_L[gi(1, 1)] = 8;
  qprint(kernel_L);
  // run!
  pic_L = dyp::filter2D(pic, kernel_L, dyp::BORDER_REPLICATE);

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
  pic_S = dyp::abs(dyp::filter2D(pic, kernel_S1, dyp::BORDER_REPLICATE)) +
          dyp::abs(dyp::filter2D(pic, kernel_S2, dyp::BORDER_REPLICATE));

  // Save Picture
  pic.imwrite("./image_out/p1.png");
  pic_L.imwrite("./image_out/p_L.png");
  pic_S.imwrite("./image_out/p_S.png");
}
/*
 * @Author: DyllanElliia
 * @Date: 2021-09-30 17:00:48
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-11 17:30:07
 * @Description:
 */

#include "../dyPicture.hpp"

int main() {
  qprint("\033[1;4;33mPicture_filter2D example:", "Author: DyllanElliia",
         "Description: How to use filter2D.\033[0m");

  dym::TimeLog t;
  // Read Picture
  dym::Picture<float, 1> pic;
  pic.imread("./image/luna.png");

  dym::Picture<float, 1> pic_L, pic_S;

  // Create Laplacian kernel
  dym::Matrix<float> kernel_L(dym::gi(3, 3), -1);
  kernel_L[dym::gi(1, 1)] = 8;
  qprint(kernel_L);
  // run!
  pic_L = dym::filter2D(pic, kernel_L, dym::BORDER_REPLICATE);

  t.record();
  t.reStart();
  // Create Roberts kernel
  dym::Matrix<float> kernel_S1(dym::gi(2, 2), []() {
    std::vector<float> v{-1, 0, 0, 1};
    return v;
  });
  qprint(kernel_S1);
  dym::Matrix<float> kernel_S2(dym::gi(2, 2), []() {
    std::vector<float> v{0, -1, 1, 0};
    return v;
  });
  qprint(kernel_S2);
  // run!
  pic_S = dym::abs(dym::filter2D(pic, kernel_S1, dym::BORDER_REPLICATE)) +
          dym::abs(dym::filter2D(pic, kernel_S2, dym::BORDER_REPLICATE));
  t.record();
  // Save Picture
  pic_L.imwrite("./image_out/p_L.png");
  pic_S.imwrite("./image_out/p_S.png");
}
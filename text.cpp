/*
 * @Author: DyllanElliia
 * @Date: 2021-09-13 16:40:59
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-06 17:01:23
 * @Description:
 */
#include "./dyPicture.hpp"

int main() {
  Picture<float, 1> pic;
  pic.imread("./example/image/luna.png");

  Matrix<float> kernel_L(gi(3, 3), -1);
  kernel_L[gi(1, 1)] = 8;
  qprint(kernel_L);

  Picture<float, 1> pic_L, pic_S;
  pic_L = dyp::filter2D(pic, kernel_L, dyp::BORDER_REPLICATE);

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

  pic_S = dyp::abs(dyp::filter2D(pic, kernel_S1, dyp::BORDER_REPLICATE)) +
          dyp::abs(dyp::filter2D(pic, kernel_S2, dyp::BORDER_REPLICATE));

  pic.imwrite("./example/image_out/asdfp1.png");
  pic_L.imwrite("./example/image_out/asdfp_L.png");
  pic_S.imwrite("./example/image_out/asdfp_S.png");
}

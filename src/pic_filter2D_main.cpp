/*
 * @Author: DyllanElliia
 * @Date: 2021-09-30 17:00:48
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-17 16:36:19
 * @Description:
 */

#include <dyMath.hpp>
#include <dyPicture.hpp>

int main() {
  qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::YELLOW);
  qprint("Picture_filter2D example:");
  qprint("Author: DyllanElliia");
  qprint("Description: How to use filter2D.");
  qp_ctrl();

  dym::TimeLog t;
  // Read Picture
  auto pic = dym::imread("./image/luna.png", float(0), dym::PIC_GRAY);

  // Create Laplacian kernel
  dym::Tensor<float> kernel_L(-1, dym::gi(3, 3));
  kernel_L[dym::gi(1, 1)] = 8;
  qprint(kernel_L);
  // run!
  auto pic_L = dym::filter2D(pic, kernel_L, dym::BORDER_REPLICATE);

  t.record();
  t.reStart();
  // Create Roberts kernel
  dym::Tensor<float> kernel_S1(dym::gi(2, 2), []() {
    std::vector<float> v{-1, 0, 0, 1};
    return v;
  });
  qprint(kernel_S1);
  dym::Tensor<float> kernel_S2(dym::gi(2, 2), []() {
    std::vector<float> v{0, -1, 1, 0};
    return v;
  });
  qprint(kernel_S2);
  // run!
  auto pic_S = dym::abs(dym::filter2D(pic, kernel_S1, dym::BORDER_REPLICATE)) +
               dym::abs(dym::filter2D(pic, kernel_S2, dym::BORDER_REPLICATE));
  t.record();
  // Save Picture
  imwrite(pic_L, "./image_out/p_L.png");
  imwrite(pic_S, "./image_out/p_S.png");
  qprint("sizeof pic: " + std::to_string(sizeof(pic_L)));
  qprint("sizeof tensor: " + std::to_string(sizeof(dym::Tensor<float>)));
}
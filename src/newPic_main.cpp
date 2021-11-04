/*
 * @Author: DyllanElliia
 * @Date: 2021-11-04 20:20:37
 * @LastEditTime: 2021-11-04 20:20:37
 * @LastEditors: DyllanElliia
 * @Description: 
 */
#include <dyPicture.hpp>

int main() {
  dym::TimeLog T;
  auto pic = dym::imread("./image/uestc.jpg", float(0), dym::PIC_RGB);

  for (int i = 100; i < 200; ++i)
    for (int j = 100; j < 200; ++j)
      pic[dym::gi(i, j, 0)] += 50, pic[dym::gi(i, j, 1)] += 50,
          pic[dym::gi(i, j, 2)] += 50;

  dym::Tensor<float> kernel_L(-1, dym::gi(3, 3));
  kernel_L[dym::gi(1, 1)] = 8;
  qprint(kernel_L);

  auto pic_l = dym::filter2D(pic, kernel_L, dym::BORDER_REPLICATE, 0.f);

  dym::imwrite(pic_l, "./image_out/newPic_l.png");
  auto ret = dym::imwrite(pic, "./image_out/newPic.jpg");
  qprint(dym::pi(pic.shape()), ret);
  return 0;
}
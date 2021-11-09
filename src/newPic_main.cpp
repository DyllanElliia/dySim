/*
 * @Author: DyllanElliia
 * @Date: 2021-11-04 20:20:37
 * @LastEditTime: 2021-11-09 10:27:24
 * @LastEditors: Please set LastEditors
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
  auto r = dym::clear(pic_l, dym::gi(0, 150, 150));
  qprint("fin", r);
  dym::Tensor<float> loc({{10, 200}, {50, 150}, {100, 50}, {200, 200}});
  loc.show();
  qprint(dym::pi(loc.shape()), loc[0], loc[1]);
  auto ans = dym::scatter(pic_l, loc, dym::gi(200, 50, 50), 3);
  qprint(ans, dym::pi(pic_l.shape()));
  dym::imwrite(pic_l, "./image_out/newPic2.jpg");
  return 0;
}
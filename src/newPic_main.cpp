/*
 * @Author: DyllanElliia
 * @Date: 2021-11-04 20:20:37
 * @LastEditTime: 2022-03-10 17:35:29
 * @LastEditors: DyllanElliia
 * @Description:
 */
#include <dyPicture.hpp>

int main() {
  dym::TimeLog T;
  auto pic = dym::imread<int, dym::PIC_RGB>("./image/uestc.jpg");

  for (int i = 100; i < 200; ++i)
    for (int j = 100; j < 200; ++j) pic[dym::gi(i, j)] += 50;

  dym::Matrix<int, 3, 3> kernel_L(-1);
  kernel_L[1][1] = 8;
  qprint(kernel_L);

  auto pic_l = dym::filter2D(pic, kernel_L, dym::BORDER_REPLICATE);

  dym::imwrite(pic_l, "./image_out/newPic_l.png");
  auto ret = dym::imwrite(pic, "./image_out/newPic.jpg");
  qprint(dym::pi(pic.shape()), ret);
  auto r = dym::clear(pic_l, dym::Vector<int, 3UL>({0, 150, 150}));
  qprint("fin", r);
  dym::Tensor<dym::Vector<Real, 2>> loc(
      {dym::Vector<Real, 2>({0.1, 0.2}), dym::Vector<Real, 2>({0.3, 0.4}),
       dym::Vector<Real, 2>({0.5, 0.6}), dym::Vector<Real, 2>({0.7, 0.8})},
      true);
  loc.show();
  qprint(dym::pi(loc.shape()), loc[0], loc[1]);
  auto ans = dym::scatter(pic_l, loc, dym::Vector<int, 3>({200, 50, 50}), 1);
  qprint(ans, dym::pi(pic_l.shape()));
  dym::imwrite(pic_l, "./image_out/newPic2.jpg");
  return 0;
}
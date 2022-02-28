/*
 * @Author: DyllanElliia
 * @Date: 2022-02-28 15:41:16
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-28 18:06:38
 * @Description:
 */
#include <dyPicture.hpp>
#include <dyGraphic.hpp>

int main(int argc, char const *argv[]) {
  dym::TimeLog T;
  auto pic = dym::imread<unsigned short, dym::PIC_RGB>("./image/luna_rgb.png");

  // pic.for_each_i([&](dym::Vector<unsigned short, 3> &e, int i, int j) {
  //   // if (i < 200)
  //   e = dym::Vector<int, 3>({255, 0, 0}).cast<unsigned short>();
  //   // else if (i < 400)
  //   //   e = dym::Vector<int, 3>({0, 255, 0}).cast<unsigned short>();
  //   // else
  //   //   e = dym::Vector<int, 3>({0, 0, 255}).cast<unsigned short>();
  // });

  // GUI part:
  dym::GUI gui("dymathTest", dym::gi(0, 100, 100));
  gui.init(pic.shape()[1], pic.shape()[0]);
  gui.update([&]() { gui.imshow(pic); });
  dym::imwrite(pic, "./image_out/guitest.png");
  return 0;
}

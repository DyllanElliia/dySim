/*
 * @Author: DyllanElliia
 * @Date: 2022-02-28 15:41:16
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-05-23 16:14:30
 * @Description:
 */
#include <dyPicture.hpp>
#include <dyGraphic.hpp>

int main(int argc, char const *argv[]) {
  dym::TimeLog T;
  auto pic = dym::imread<unsigned char, dym::PIC_RGB>("./image/luna_rgb.png");
  // GUI part:
  dym::GUI gui("dymathTest", dym::gi(0, 100, 100));
  gui.init(pic.shape()[0], pic.shape()[1]);
  // int i=0;
  gui.update([&]() {
    pic += 3;
    gui.imshow(pic);
  });
  dym::imwrite(pic, "./image_out/guitest.png");
  return 0;
}

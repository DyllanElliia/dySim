/*
 * @Author: DyllanElliia
 * @Date: 2021-09-13 16:40:59
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-08 17:27:17
 * @Description:
 */
#include "./dyPicture.hpp"

int main() {
  dym::TimeLog t;
  dym::Picture<float, 1> pic;
  pic.imread("./example/image/luna.png");

  dym::Picture<float, 1> pic_I, pic_L, pic_D;

  pic_I = 255 - pic;

  pic_L = 2 * pic;

  pic_D = 0.5 * pic;

  pic_I.imwrite("./example/image_out/p_I.png");
  pic_L.imwrite("./example/image_out/p_L.png");
  pic_D.imwrite("./example/image_out/p_D.png");
}

/*
 * @Author: DyllanElliia
 * @Date: 2021-10-08 17:03:31
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-08 17:07:04
 * @Description: How to use Picture.
 */
#include "../dyPicture.hpp"

int main() {
  qprint("\033[1;4;33mPicture example:", "Author: DyllanElliia",
         "Description: How to use Picture.\033[0m");

  dym::Picture<float, 1> pic;
  pic.imread("./image/luna.png");

  dym::Picture<float, 1> pic_I, pic_L, pic_D, pic_A;

  pic_I = 255 - pic;

  pic_L = 2 * pic;

  pic_D = 0.5 * pic;

  pic_A = 0.5 * pic + 0.5 * pic;

  pic_I.imwrite("./image_out/p_Inverse.png");
  pic_L.imwrite("./image_out/p_Light.png");
  pic_D.imwrite("./image_out/p_Dark.png");
  pic_A.imwrite("./image_out/p_Add.png");
}
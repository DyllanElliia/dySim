/*
 * @Author: DyllanElliia
 * @Date: 2022-02-28 15:41:16
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-05-23 16:14:30
 * @Description:
 */
#include <dyGraphic.hpp>
#include <dyPicture.hpp>

// int main(int argc, char const *argv[]) {
//   dym::TimeLog T;
//   auto pic = dym::imread<unsigned char,
//   dym::PIC_RGB>("./image/luna_rgb.png");
//   // GUI part:
//   dym::GUI gui("dymathTest", dym::gi(0, 100, 100));
//   gui.init(pic.shape()[0], pic.shape()[1]);
//   pic.for_each_p([&](dym::Vector<unsigned char, 3> &e, int i,
//                      int j) { e[0] += i, e[2] += j; },
//                  {15, 15});
//   // int i=0;
//   gui.update([&]() {
//     // pic += 3;
//     pic.for_each_p(
//         [&](dym::Vector<unsigned char, 3> &e, int i, int j) { e += 3; },
//         {15, 15});
//     gui.imshow(pic);
//   });
//   dym::imwrite(pic, "./image_out/guitest.png");
//   return 0;
// }

int main(int argc, char const *argv[]) {
  dym::TimeLog T;
  dym::Tensor<dym::Vector<unsigned char, 3>> pic(0, dym::gi(420, 236));
  // GUI part:
  dym::GUI gui("dymathTest", dym::gi(0, 100, 100));
  gui.init(pic.shape()[0], pic.shape()[1]);
  // pic.for_each_p([&](dym::Vector<unsigned char, 3> &e, int i,
  //                    int j) { e[0] += i, e[2] += j; },
  //                {15, 15});
  // int i=0;
  gui.update([&]() {
    // pic += 3;
    pic.for_each_p(
        [&](dym::Vector<unsigned char, 3> &e, int i, int j) {
          auto p = dym::Vector2{Real(i), Real(j)} / dym::Vector2{420, 236};
          auto sp = p - 0.5;
          dym::Vector3 color{.2, .7, 1.};
          dym::Vector2 c{0.4 + 0.1 * dym::cos(T.getRecord() * 0.3),
                         0.4 + 0.1 * dym::sin(T.getRecord() * 0.2)};
          auto p1 = sp, p2 = sp;
          for (int i = 0; i < 15; ++i) {
            p1 = dym::Vector2{dym::pow(p2.x(), 2.0) - dym::pow(p2.y(), 2.0),
                              2.0 * p2.x() * p2.y()};
            p2 = p1 + c;
          }
          auto fp = p2;
          auto modx = (fmod(fp.x(), 0.05) > 0.04) ? 0.0 : 1.0;
          auto mody = (fmod(fp.y(), 0.05) > 0.04) ? 0.0 : 1.0;
          color *= modx;
          color *= mody;
          e = dym::clamp(color, 0., 255.).cast<unsigned char>();
        },
        {50, 50});
    gui.imshow(pic);
  });
  dym::imwrite(pic, "./image_out/guitest.png");
  return 0;
}
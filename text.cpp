/*
 * @Author: DyllanElliia
 * @Date: 2021-09-13 16:40:59
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-09-24 19:39:34
 * @Description:
 */
#include "./dyMath.hpp"
#include "./dyPicture.hpp"

int main() {
  // std::vector<short> v{0, 4, 6};
  // Pixel<short, 3> a(v);
  // Pixel<short, 3> b({4, 2, 3});
  // std::cout << a << std::endl;
  // std::cout << b << std::endl;
  // // a = b;
  // // std::cout << a << std::endl;
  // Pixel<short, 3> c = a;
  // std::cout << c << std::endl;
  // (a / b).show();
  // c = c - 2;
  // std::cout << c << std::endl;
  // std::cout << c[1] << std::endl;
  // std::cout << "here1" << std::endl;
  // c = 0;
  // std::cout << c << std::endl;

  Picture<float, 3> pic;
  pic.imread("./example/image/uestc.png");
  std::cout << pi(pic.shape()) << std::endl;
  // pic.show();
  //   std::cout << pic[gi(0, 0)] << std::endl;
  //   std::cout << (pic[gi(100, 0)])[0]<< std::endl;
  pic[gi(100, 0)].text();
  pic.pic = 3 * pic.pic;
  //   std::cout << pic[gi(0, 0)] << std::endl;
  //   std::cout << (pic[gi(100, 0)])[0] << std::endl;
  pic.imwrite("./example/image_out/asdf.jpg");
}

/*
 * @Author: DyllanElliia
 * @Date: 2021-09-13 16:40:59
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-09-23 17:31:43
 * @Description:
 */
#include "./dyMath.hpp"
#include "./dyPicture.hpp"

int main() {
  // std::vector<short> v{0, 2, 3};
  // Pixel<short, 3> a(v);
  // Pixel<short, 3> b({4, 2, 3});
  // Pixel<short, 3> c = a / b;
  // std::cout << c << std::endl;
  // std::cout << c[1] << std::endl;
  // std::cout << "here1" << std::endl;
  // c = 0;
  // std::cout << c << std::endl;
  Picture<short, 3> pic;
  std::cout << "here2" << std::endl;
  pic.imread("./example/image/uestc.jpg");
  std::cout << "here\n" << pi(pic.shape()) << std::endl;
  // pic.show();
  std::cout << pic[gi(0, 0)] << std::endl;
  std::cout << (pic[gi(100, 0)])[0] << std::endl;
  pic[gi(100, 0)].text();
  // pic.pic = pic.pic - 100;
  pic.imwrite("./example/image/asdf.png");
}

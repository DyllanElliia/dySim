/*
 * @Author: DyllanElliia
 * @Date: 2021-09-13 16:50:00
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-19 19:23:48
 * @Description:
 */
#pragma once

// !!!Abandon class Picture
// #include "src/picture.hpp"
// #include "./matrix.hpp"
#include "src/tensor.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include "../../tools/str_hash.hpp"

namespace dym {
enum BorderType { BORDER_CONSTANT = 1, BORDER_REFLECT, BORDER_REPLICATE };

// template <class InputType, class KernelType, int color_size>
// Picture<InputType, color_size>
// filter2D(Picture<InputType, color_size> &in, KernelType &kernel,
//          BorderType border = BORDER_CONSTANT,
//          Tuples<InputType, color_size> DefaultColor =
//              Tuples<InputType, color_size>(0)) {
//   Index vShape = in.shape(), vBorder1 = vShape - gi(1, 1, 0);
//   auto &vShapeX = vShape[0], &vShapeY = vShape[1];
//   auto &vBorder1X = vBorder1[0], &vBorder1Y = vBorder1[1];
//   std::map<BorderType, std::function<Tuples<InputType, color_size>(int,
//   int)>>
//       gvm = {{BORDER_CONSTANT,
//               [&DefaultColor, &vBorder1X, &vBorder1Y, &in](int x, int y) {
//                 if (x < 0 || x > vBorder1X || y < 0 || y > vBorder1Y)
//                   return DefaultColor;
//                 return in[gi(x, y)];
//               }},
//              {BORDER_REFLECT,
//               [&DefaultColor, &vBorder1X, &vBorder1Y, &in](int x, int y) {
//                 return in[gi((x < 0 ? 0 : (x > vBorder1X ? vBorder1X : x)),
//                              (y < 0 ? 0 : (y > vBorder1Y ? vBorder1Y : y)))];
//               }},
//              {BORDER_REPLICATE,
//               [&DefaultColor, &vBorder1X, &vBorder1Y, &in](int x, int y) {
//                 while (x < 0 || x > vBorder1X)
//                   x = x < 0 ? -x : (x > vBorder1X ? 2 * vBorder1X - x : x);
//                 while (y < 0 || y > vBorder1Y)
//                   y = y < 0 ? -y : (y > vBorder1Y ? 2 * vBorder1Y - y : y);
//                 return in[gi(x, y)];
//               }}};
//   const auto &gv = gvm[border];

//   Picture<InputType, color_size> result(gi(vShapeX, vShapeY));

//   Index kShape = kernel.shape();
//   auto &kShapeX = kShape[0], &kShapeY = kShape[1];
//   auto kCenterX = kShapeX >> 1, kCenterY = kShapeY >> 1;
//   auto kSize = kShapeX * kShapeY;
//   int kSXY = kShapeX * kShapeY;
//   std::vector<int> kx(kSXY, 0), ky(kSXY, 0);
//   std::vector<int> kxi(kSXY, -kCenterX), kyi(kSXY, -kCenterY);
//   int cnt = 0;
//   for (int ki = 0; ki < kShapeX; ++ki)
//     for (int kj = 0; kj < kShapeY; ++kj) {
//       kx[cnt] = ki;
//       ky[cnt] = kj;
//       kxi[cnt] += ki;
//       kyi[cnt] += kj;
//       ++cnt;
//     }
//   auto forI = [&result, &gv, &kernel, &vShapeY, &kxi, &kyi, &kx, &ky,
//                &kSXY](const unsigned int ib, const unsigned int ie) {
//     // qprint_nlb(ib, ie);
//     for (unsigned int i = ib; i < ie; ++i)
//       for (int j = 0; j < vShapeY; ++j) {
//         auto &rij = result[gi(i, j)];
//         rij = 0;
//         for (int k = 0; k < kSXY; ++k)
//           rij += gv(i + kxi[k], j + kyi[k]) * kernel[gi(kx[k], ky[k])];
//       }
//   };
//   const unsigned int t_num = std::thread::hardware_concurrency() / 3;
//   const unsigned int t_step = (vShapeX + t_num) / t_num;
//   qprint(t_num, t_step);
//   std::vector<std::thread> t_pool;
//   for (unsigned int i = 0; i < t_num; ++i) {
//     unsigned int ib = i * t_step, ie = (i + 1) * t_step;
//     if (ie > vShapeX)
//       ie = vShapeX;
//     t_pool.push_back(std::thread(forI, ib, ie));
//   }
//   std::for_each(t_pool.begin(), t_pool.begin() + t_pool.size(),
//                 [](std::thread &t) { t.join(); });
//   // for (int i = 0; i < vShapeX; ++i)
//   //   for (int j = 0; j < vShapeY; ++j) {
//   //     auto &rij = result[gi(i, j)];
//   //     rij = 0;
//   //     for (int k = 0; k < kSXY; ++k)
//   //       rij += gv(i + kxi[k], j + kyi[k]) * kernel[gi(kx[k], ky[k])];
//   //   }

//   return result;
// }

template <class InputType, class KernelType>
Tensor<InputType> filter2D(Tensor<InputType> &in, Tensor<KernelType> &kernel,
                           BorderType border = BORDER_CONSTANT,
                           InputType DefaultColor = InputType(0)) {
  Index vShape = in.shape(), vBorder1 = vShape - gi(1, 1, 0);
  auto &color_size = vShape[2];
  auto &vShapeX = vShape[0], &vShapeY = vShape[1];
  auto &vBorder1X = vBorder1[0], &vBorder1Y = vBorder1[1];
  std::map<BorderType, std::function<InputType(int, int, int)>> gvm = {
      {BORDER_CONSTANT,
       [&DefaultColor, &vBorder1X, &vBorder1Y, &in](int x, int y, int k) {
         if (x < 0 || x > vBorder1X || y < 0 || y > vBorder1Y)
           return DefaultColor;
         return in[gi(x, y, k)];
       }},
      {BORDER_REFLECT,
       [&DefaultColor, &vBorder1X, &vBorder1Y, &in](int x, int y, int k) {
         return in[gi((x < 0 ? 0 : (x > vBorder1X ? vBorder1X : x)),
                      (y < 0 ? 0 : (y > vBorder1Y ? vBorder1Y : y)), k)];
       }},
      {BORDER_REPLICATE,
       [&DefaultColor, &vBorder1X, &vBorder1Y, &in](int x, int y, int k) {
         while (x < 0 || x > vBorder1X)
           x = x < 0 ? -x : (x > vBorder1X ? 2 * vBorder1X - x : x);
         while (y < 0 || y > vBorder1Y)
           y = y < 0 ? -y : (y > vBorder1Y ? 2 * vBorder1Y - y : y);
         return in[gi(x, y, k)];
       }}};
  const auto &gv = gvm[border];

  Tensor<InputType> result(0, vShape);

  Index kShape = kernel.shape();
  auto &kShapeX = kShape[0], &kShapeY = kShape[1];
  auto kCenterX = kShapeX >> 1, kCenterY = kShapeY >> 1;
  auto kSize = kShapeX * kShapeY;
  int kSXY = kShapeX * kShapeY;
  std::vector<int> kx(kSXY, 0), ky(kSXY, 0);
  std::vector<int> kxi(kSXY, -kCenterX), kyi(kSXY, -kCenterY);
  int cnt = 0;
  for (int ki = 0; ki < kShapeX; ++ki)
    for (int kj = 0; kj < kShapeY; ++kj) {
      kx[cnt] = ki;
      ky[cnt] = kj;
      kxi[cnt] += ki;
      kyi[cnt] += kj;
      ++cnt;
    }

  result.for_each([&result, &gv, &kernel, &vShapeY, &kxi, &kyi, &kx, &ky,
                   &kSXY](InputType &e, int i, int j, int k) {
    auto &rij = result[gi(i, j, k)];
    for (int l = 0; l < kSXY; ++l)
      rij += gv(i + kxi[l], j + kyi[l], k) * kernel[gi(kx[l], ky[l])];
  });

  return result;
}

// template <class InputType, int color_size>
// Picture<InputType, color_size> abs(Picture<InputType, color_size> in) {
//   Picture<InputType, color_size> result(in);
//   result.for_each([](Tuples<InputType, color_size> &i) {
//     for (int j = 0; j < color_size; ++j)
//       if (i[j] < 0)
//         i[j] = -i[j];
//   });

//   return result;
// }

const int PIC_GRAY = 1;
const int PIC_RGB = 3;

template <typename Type = short>
Tensor<Type> imread(std::string picPath, Type ValueType = Type(0),
                    int color_size = PIC_RGB) {
  int channel = 0;
  int x, y;
  unsigned char *data = stbi_load(picPath.c_str(), &x, &y, &channel, 0);
  try {
    if (channel > color_size)
      throw "\033[1;31mimread warning: Picture's color_size must be larger "
            "than picture's channel!\033[0m";
    if (data == nullptr)
      throw "\033[1;31mimread error: Image reading failure.\033[0m";
  } catch (const char *str) {
    std::cerr << str << '\n';
    exit(EXIT_FAILURE);
  }
  Tensor<Type> result(0, gi(y, x, color_size));
  int xc = x * channel;
  if (channel < color_size)
    result.for_each(
        [&data, &x](Type &e, int i, int j) { e = (Type)data[i * x + j]; });
  else
    result.for_each([&data, &xc, &channel](Type &e, int i, int j, int k) {
      e = (Type)data[i * xc + j * channel + k];
    });
  stbi_image_free(data);
  return result;
}

template <typename Type = short>
Tensor<Type> imwrite(Tensor<Type> &pic, std::string picPath) {
  if (pic.shape().size() != 3)
    return -1;
  Index size_ = pic.shape();
  auto &color_size = size_[2];
  int &x = size_[1], &y = size_[0];
  int xc = x * color_size, size_i = y * xc;
  // std::cout << "here " << size_i << std::endl;
  unsigned char *data = new unsigned char[size_i];
  pic.for_each([&data, &xc, &color_size](Type &e, int i, int j, int k) {
    auto &datai = data[i * xc + j * color_size + k];
    auto pici = e;
    if (pici < 0)
      pici = 0;
    if (pici > 255)
      pici = 255;
    datai = pici;
  });
  int fne_r = picPath.rfind('.');
  if (fne_r == std::string::npos || fne_r == picPath.size() - 1 ||
      picPath[fne_r + 1] == '/' || picPath[fne_r + 1] == '\\') {
    std::cout << "\033[1;31mimwrite error: failed to write picture to "
                 "\033[0m\033[4;33m\""
              << picPath
              << "\"\033[0m\033[1;31m. without picture's type e.g. "
                 "~/pic.jpg\033[0m"
              << std::endl;
    delete[] data;
    return -1;
  }
  int return_ = -1;
  std::string picPath_end = picPath.substr(fne_r + 1);

  switch (hash_(picPath_end.c_str())) {
  case hash_compile_time("jpg"):
    return_ = stbi_write_jpg(picPath.c_str(), x, y, color_size, data, 0);
    break;
  case hash_compile_time("png"):
    return_ = stbi_write_png(picPath.c_str(), x, y, color_size, data, 0);
    break;
  case hash_compile_time("bmp"):
    return_ = stbi_write_bmp(picPath.c_str(), x, y, color_size, data);
    break;
  case hash_compile_time("tga"):
    return_ = stbi_write_tga(picPath.c_str(), x, y, color_size, data);
    break;
  case hash_compile_time(""):
    std::cout << "\033[1;31mimwrite error: please specify a file type for "
                 "writing.\033[0m"
              << std::endl;
    break;
  default:
    std::cout << "\033[1;31mimwrite error: dyPicture does not support write " +
                     picPath_end + ".\033[0m"
              << std::endl;
  }
  delete[] data;
  return return_;
}

} // namespace dym

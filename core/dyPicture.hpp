/*
 * @Author: DyllanElliia
 * @Date: 2021-09-13 16:50:00
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-26 17:22:54
 * @Description:
 */
#pragma once

// !!!Abandon class Picture & Matrix
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

template <typename ValueType, typename colorType>
bool clear(Tensor<ValueType> &pic, colorType const color_default) {
  const auto color_size = color_default.size();
  const auto &pic_color_size = (pic.shape().size() == 3 ? pic.shape()[2] : 1);
  if (pic_color_size > color_size) return false;
  // qprint(pic_color_size, color_size);
  pic.for_each_i([&color_default, &pic_color_size](ValueType &e, int i) {
    e = color_default[i % pic_color_size];
  });
  return true;
}

template <typename ValueType, typename VertexType>
int scatter(Tensor<ValueType> &pic, Tensor<VertexType> &loc,
            const Index<int> color_default, int radius = 1) {
  const int pic_color_size = (pic.shape().size() == 3 ? pic.shape()[2] : 1);
  const int color_size = color_default.size();
  if (pic_color_size > color_size) return -1;
  // qprint(pic_color_size, color_size);
  const Index locShape = loc.shape();
  const auto &vec_num = locShape[0], &vec_d = locShape[1];
  if (vec_d != 2) return -1;
  const Index picShape = pic.shape();
  const auto &picx = picShape[0], &picy = picShape[1];

  // std::array<int, 2> pos;
  int r2 = 4 * radius * radius;
  std::vector<int> xx(r2), yy(r2);
  for (int x = -radius, index = 0; x < radius; ++x)
    for (int y = -radius; y < radius; ++y, ++index)
      xx[index] = x, yy[index] = y;

  int ans = 0;

  loc.for_each([&](VertexType *pos, int i) {
    int ans_i = 0;
    for (int xyi = 0; xyi < r2; ++xyi) {
      int xi = pos[0] + xx[xyi], yi = pos[1] + yy[xyi];
      if (xi < 0 || xi >= picx || yi < 0 || yi >= picy) continue;
      if (pic_color_size == 1) {
        pic[gi(xi, yi)] = color_default[0];
        ++ans_i;
      } else
        for (int ci = 0; ci < pic_color_size; ++ci) {
          pic[gi(xi, yi, ci)] = color_default[ci];
          ++ans_i;
        }
    }
    ans += ans_i;
  });
  // for (int i = 0; i < vec_num; ++i) {
  //   int pbegin = i * vec_d;
  //   pos[0] = loc[pbegin], pos[1] = loc[pbegin + 1];
  //   for (int xyi = 0; xyi < r2; ++xyi) {
  //     int xi = pos[0] + xx[xyi], yi = pos[1] + yy[xyi];
  //     if (xi < 0 || xi >= picx || yi < 0 || yi >= picy) continue;
  //     if (pic_color_size == 1) {
  //       pic[gi(xi, yi)] = color_default[0];
  //       ++ans;
  //     } else
  //       for (int ci = 0; ci < pic_color_size; ++ci) {
  //         pic[gi(xi, yi, ci)] = color_default[ci];
  //         ++ans;
  //       }
  //   }
  // }
  return ans;
}

template <class InputType, class KernelType>
Tensor<InputType> filter2D(Tensor<InputType> &in, Tensor<KernelType> &kernel,
                           BorderType border = BORDER_CONSTANT,
                           InputType DefaultColor = InputType(0)) {
  Index<int> vShape = in.shape(), vBorder1 = vShape - gi(1, 1, 0);
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

  result.for_each_i([&result, &gv, &kernel, &vShapeY, &kxi, &kyi, &kx, &ky,
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
//   result.for_each_i([](Tuples<InputType, color_size> &i) {
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
    result.for_each_i(
        [&data, &x](Type &e, int i, int j) { e = (Type)data[i * x + j]; });
  else
    result.for_each_i([&data, &xc, &channel](Type &e, int i, int j, int k) {
      e = (Type)data[i * xc + j * channel + k];
    });
  stbi_image_free(data);
  return result;
}

template <typename Type = short>
int imwrite(Tensor<Type> &pic, std::string picPath) {
  Index size_ = pic.shape();
  if (size_.size() != 3) return -1;
  auto &color_size = size_[2];
  int &x = size_[1], &y = size_[0];
  int xc = x * color_size, size_i = y * xc;
  // std::cout << "here " << size_i << std::endl;
  unsigned char *data = new unsigned char[size_i];
  pic.for_each_i([&data, &xc, &color_size](Type &e, int i, int j, int k) {
    auto &datai = data[i * xc + j * color_size + k];
    auto pici = e;
    if (pici < 0) pici = 0;
    if (pici > 255) pici = 255;
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
      std::cout
          << "\033[1;31mimwrite error: dyPicture does not support write " +
                 picPath_end + ".\033[0m"
          << std::endl;
  }
  delete[] data;
  return return_;
}

}  // namespace dym

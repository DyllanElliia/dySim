/*
 * @Author: DyllanElliia
 * @Date: 2021-09-13 16:50:00
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-28 16:20:15
 * @Description:
 */
#pragma once

#include "src/tensor.hpp"
#include "src/matrix.hpp"
#include <string>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#endif

#include "../../tools/str_hash.hpp"

namespace dym {
enum BorderType { BORDER_CONSTANT = 1, BORDER_REFLECT, BORDER_REPLICATE };

template <class ValueType, std::size_t color_size>
bool clear(Tensor<Vector<ValueType, color_size>> &pic,
           const Vector<ValueType, color_size> &color_default) {
  pic = color_default;
  return true;
}

template <typename ValueType, typename VertexType, std::size_t color_size>
int scatter(Tensor<Vector<ValueType, color_size>> &pic,
            Tensor<Vector<VertexType, 2>> &loc,
            const Vector<ValueType, color_size> color_default,
            const int radius = 1) {
  const Index locShape = loc.shape();
  const auto &vec_num = locShape[0], &vec_d = locShape[1];
  if (vec_d != 1) return -1;
  const Index picShape = pic.shape();
  const auto &picx = picShape[0], &picy = picShape[1];
  const float fpicx = picx, fpicy = picy;

  // std::array<int, 2> pos;
  const int r2 = 4 * radius * radius;
  std::vector<int> xx(r2), yy(r2);
  for (int x = -radius, index = 0; x < radius; ++x)
    for (int y = -radius; y < radius; ++y, ++index)
      xx[index] = x, yy[index] = y;

  int ans = 0;

  loc.for_each_i([&](Vector<VertexType, 2> &pos, int i) {
    int ans_i = 0;
    for (int xyi = 0; xyi < r2; ++xyi) {
      // int xi = pos[0] + xx[xyi], yi = pos[1] + yy[xyi];
      int xi = (picx - fpicx * pos[1]) + xx[xyi],
          yi = (pos[0] * fpicy) + yy[xyi];
      // qprint(std::to_string(i) + " (" + std::to_string(pos[0]) + " " +
      //        std::to_string(pos[1]) + ")->(" + std::to_string(xi) + " " +
      //        std::to_string(yi) + ")");
      if (xi < 0 || xi >= picx || yi < 0 || yi >= picy) continue;
      pic[gi(xi, yi)] = color_default, ++ans_i;
    }
    ans += ans_i;
  });

  return ans;
}

template <class InputType, class KernelType, std::size_t color_size,
          std::size_t kernel_m, std::size_t kernel_n>
Tensor<Vector<InputType, color_size>> filter2D(
    Tensor<Vector<InputType, color_size>> &in,
    Matrix<KernelType, kernel_m, kernel_n> &kernel,
    BorderType border = BORDER_CONSTANT,
    Vector<InputType, color_size> DefaultColor =
        Vector<InputType, color_size>(0)) {
  Index<int> vShape = in.shape(), vBorder1 = vShape - gi(1, 1);
  auto &vShapeX = vShape[0], &vShapeY = vShape[1];
  auto &vBorder1X = vBorder1[0], &vBorder1Y = vBorder1[1];
  std::map<BorderType, std::function<Vector<InputType, color_size>(int, int)>>
      gvm = {{BORDER_CONSTANT,
              [&DefaultColor, &vBorder1X, &vBorder1Y, &in](int x, int y) {
                if (x < 0 || x > vBorder1X || y < 0 || y > vBorder1Y)
                  return DefaultColor;
                return in[gi(x, y)];
              }},
             {BORDER_REFLECT,
              [&DefaultColor, &vBorder1X, &vBorder1Y, &in](int x, int y) {
                return in[gi((x < 0 ? 0 : (x > vBorder1X ? vBorder1X : x)),
                             (y < 0 ? 0 : (y > vBorder1Y ? vBorder1Y : y)))];
              }},
             {BORDER_REPLICATE,
              [&DefaultColor, &vBorder1X, &vBorder1Y, &in](int x, int y) {
                while (x < 0 || x > vBorder1X)
                  x = x < 0 ? -x : (x > vBorder1X ? 2 * vBorder1X - x : x);
                while (y < 0 || y > vBorder1Y)
                  y = y < 0 ? -y : (y > vBorder1Y ? 2 * vBorder1Y - y : y);
                return in[gi(x, y)];
              }}};
  const auto &gv = gvm[border];

  Tensor<Vector<InputType, color_size>> result(0, vShape);

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
                     &kSXY](Vector<InputType, color_size> &e, int i, int j) {
    auto &rij = result[gi(i, j)];
    for (int l = 0; l < kSXY; ++l)
      rij += gv(i + kxi[l], j + kyi[l]) * kernel[kx[l]][ky[l]];
  });
  return result;
}

const std::size_t PIC_GRAY = 1;
const std::size_t PIC_RGB = 3;

template <typename Type = short, std::size_t color_size = PIC_RGB>
Tensor<Vector<Type, color_size>> imread(std::string picPath) {
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
  Tensor<Vector<Type, color_size>> result(0, gi(y, x));
  int xc = x * channel;
  if (channel < color_size)
    result.for_each_i([&data, &x](Vector<Type, color_size> &e, int i, int j) {
      e = (Type)data[i * x + j];
    });
  else
    result.for_each_i(
        [&data, &xc, &channel](Vector<Type, color_size> &e, int i, int j) {
          for (int k = 0; k < color_size; ++k)
            e[k] = (Type)data[i * xc + j * channel + k];
        });
  stbi_image_free(data);
  return result;
}

template <typename Type = short, std::size_t color_size>
int imwrite(Tensor<Vector<Type, color_size>> &pic, std::string picPath) {
  Index size_ = pic.shape();
  if (size_.size() != 2) return -1;
  int &x = size_[1], &y = size_[0];
  int xc = x * color_size, size_i = y * xc;
  // std::cout << "here " << size_i << std::endl;
  unsigned char *data = new unsigned char[size_i];
  pic.for_each_i([&data, &xc](Vector<Type, color_size> &e, int i, int j) {
    for (int k = 0; k < color_size; ++k) {
      auto &datai = data[i * xc + j * color_size + k];
      auto pici = e[k];
      if (pici < 0) pici = 0;
      if (pici > 255) pici = 255;
      datai = pici;
    }
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

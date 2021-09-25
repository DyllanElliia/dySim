#pragma once

#include "./dyMath.hpp"
#include "./pixel.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"

#include "./tools/str_hash.h"

template <typename T = short, int color_size = 3> class Picture {
private:
  using ValueType = T;
  Matrix<Pixel<ValueType, color_size>> pic;
  int channel;

  void imread_(std::string filename) {
    int x, y;
    unsigned char *data =
        stbi_load(filename.c_str(), &x, &y, &channel, color_size);
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
    pic.reShape(gi(x, y));
    int yc = y * channel;
    for (int j = 0; j < y; ++j)
      for (int i = 0; i < x; ++i)
        for (int k = 0; k < color_size; ++k) {
          // std::cout << (ValueType)data[i * yc + j * color_size + k]
          //           << std::endl;
          pic[gi(i, j)][k] = (ValueType)data[i * yc + j * color_size + k];
          // std::cout << pic[gi(i, j)][k] << std::endl;
        }
    stbi_image_free(data);
  }

  int imwrite_(std::string filename) {
    Index size_ = pic.shape();
    int &x = size_[0], &y = size_[1];
    int yc = y * channel, size_i = x * yc;
    // std::cout << "here " << size_i << std::endl;
    unsigned char *data = new unsigned char[size_i];

    for (int j = 0; j < y; ++j)
      for (int i = 0; i < x; ++i)
        for (int k = 0; k < color_size; ++k) {
          // std::cout << i << " " << j << " " << k << " " << pic[gi(i, j)][k]
          //           << " " << i * yc + j * color_size + k << std::endl;
          auto &datai = data[i * yc + j * color_size + k];
          auto pici = pic[gi(i, j)][k];
          if (pici < 0)
            pici = 0;
          if (pici > 255)
            pici = 255;
          datai = pici;
        }
    // std::cout << "here" << std::endl;
    int fne_r = filename.rfind('.');
    if (fne_r == std::string::npos || fne_r == filename.size() - 1 ||
        filename[fne_r + 1] == '/' || filename[fne_r + 1] == '\\') {
      std::cout << "\033[1;31mimwrite error: failed to write picture to "
                   "\033[0m\033[4;33m\""
                << filename
                << "\"\033[0m\033[1;31m. without picture's type e.g. "
                   "~/pic.jpg\033[0m"
                << std::endl;
      delete[] data;
      return -1;
    }
    int return_ = -1;
    std::string filename_end = filename.substr(fne_r + 1);

    switch (hash_(filename_end.c_str())) {
    case hash_compile_time("jpg"):
      return_ = stbi_write_jpg(filename.c_str(), x, y, channel, data, 0);
      break;
    case hash_compile_time("png"):
      return_ = stbi_write_png(filename.c_str(), x, y, channel, data, 0);
      break;
    case hash_compile_time("bmp"):
      return_ = stbi_write_bmp(filename.c_str(), x, y, channel, data);
      break;
    case hash_compile_time("tga"):
      return_ = stbi_write_tga(filename.c_str(), x, y, channel, data);
      break;
    case hash_compile_time(""):
      std::cout << "\033[1;31mimwrite error: please specify a file type for "
                   "writing.\033[0m"
                << std::endl;
      break;
    default:
      std::cout
          << "\033[1;31mimwrite error: dyPicture does not support write " +
                 filename_end + ".\033[0m"
          << std::endl;
    }
    delete[] data;
    return return_;
  }

public:
  Picture() {}
  ~Picture() {}

  inline int getChannel() const { return channel; }
  inline void imread(std::string filename) { imread_(filename); }
  inline int imwrite(std::string filename) { return imwrite_(filename); }
  inline bool show(const std::string &colTabStr = "|   ") {
    return pic.show(colTabStr);
  }
  Index shape() {
    Index i_ = pic.shape();
    i_.push_back(getChannel());
    return i_;
  }

  Pixel<ValueType, color_size> &operator[](const Index &index_) {
    return pic[index_];
  }

  friend Picture operator+(const Picture &first, const Picture &second) {
    Picture result;
    result.pic = first.pic + second.pic;
    return result;
  }

  /* 	Picture operator+(const ValueType& second) {
          Picture result(*this);
          for (auto& i : result.a) i = i + second;
          return result;
  } */

  friend Picture operator+(const ValueType &first, Picture &second) {
    Picture result;
    result.pic = first + second.pic;
    return result;
  }

  friend Picture operator+(Picture &first, const ValueType &second) {
    Picture result;
    result.pic = first.pic + second;
    return result;
  }

  friend Picture operator-(const Picture &first, const Picture &second) {
    Picture result;
    result.pic = first.pic - second.pic;
    return result;
  }

  friend Picture operator-(const ValueType &first, Picture &second) {
    Picture result;
    result.pic = first - second.pic;
    return result;
  }

  friend Picture operator-(Picture &first, const ValueType &second) {
    Picture result;
    result.pic = first.pic - second;
    return result;
  }

  Picture operator*(const ValueType &second) {
    Picture result;
    result.pic = pic * second;
    return result;
  }

  friend Picture operator*(const ValueType &first, Picture &second) {
    Picture result;
    result.pic = first * second.pic;
    return result;
  }

  Picture operator/(const ValueType &second) {
    Picture result;
    result.pic = pic / second;
    return result;
  }

  friend Picture operator/(const ValueType &first, Picture &second) {
    Picture result;
    result.pic = first / second.pic;
    return result;
  }

  friend Picture operator*(const Picture &first, const Picture &second) {
    Picture result;
    result.pic = first.pic * second.pic;
    return result;
  }

  Picture &operator=(const Picture &in) {
    pic = in.pic;
    return *this;
  }

  void text() {
    // std::cout << pic.end() << std::endl;
  }
};

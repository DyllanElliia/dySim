#pragma once

// #include "./dyMath.hpp"
#include "./tensor.hpp"
#include "./tuples.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"

#include "tools/str_hash.hpp"
namespace dym {

template <typename T = short, int color_size = 3> class Picture {
private:
  using ValueType = T;
  Tensor<Tuples<ValueType, color_size>> pic;

  void imread_(std::string filename) {
    int channel = 0;
    int x, y;
    unsigned char *data = stbi_load(filename.c_str(), &x, &y, &channel, 0);
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
    // qprint_nlb(x, y, pi(pic.shape()));
    int xc = x * channel;
    // int rEnd = x * y;
    // std::cout << channel << std::endl;
    // for (int i = 0; i < y; ++i)
    //   for (int j = 0; j < x; ++j)
    //     data[i * x + j] = 255;
    if (channel < color_size)
      // for (int i = 0; i < rEnd; ++i)
      //   for (int k = 0; k < color_size; ++k) {
      //     pic[i][k] = (ValueType)data[i];
      //   }
      for (int i = 0; i < y; ++i)
        for (int j = 0; j < x; ++j)
          for (int k = 0; k < color_size; ++k) {
            pic[gi(j, i)][k] = (ValueType)data[i * x + j];
          }
    else
      // for (int i = 0; i < rEnd; ++i)
      //   for (int k = 0; k < color_size; ++k) {
      //     pic[i][k] = (ValueType)data[i * channel + k];
      //   }
      for (int i = 0; i < y; ++i)
        for (int j = 0; j < x; ++j)
          for (int k = 0; k < color_size; ++k) {
            pic[gi(j, i)][k] = (ValueType)data[i * xc + j * channel + k];
          }
    stbi_image_free(data);
  }

  int imwrite_(std::string filename) {
    Index size_ = pic.shape();
    int &x = size_[0], &y = size_[1];
    int xc = x * color_size, size_i = y * xc;
    // std::cout << "here " << size_i << std::endl;
    unsigned char *data = new unsigned char[size_i];

    for (int i = 0; i < y; ++i)
      for (int j = 0; j < x; ++j)
        for (int k = 0; k < color_size; ++k) {
          // std::cout << i << " " << j << " " << k << " " << pic[gi(i, j)][k]
          //           << " " << i * yc + j * color_size + k << std::endl;
          auto &datai = data[i * xc + j * color_size + k];
          auto pici = pic[gi(j, i)][k];
          if (pici < 0)
            pici = 0;
          if (pici > 255)
            pici = 255;
          datai = pici;
        }

    // for (int j = 0; j < y; ++j) {
    //   auto &datai = data[j * yc + j * color_size];
    //   // if (pici < 0)
    //   //   pici = 0;
    //   // if (pici > 255)
    //   //   pici = 255;
    //   datai = 255;
    // }
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
      return_ = stbi_write_jpg(filename.c_str(), x, y, color_size, data, 0);
      break;
    case hash_compile_time("png"):
      return_ = stbi_write_png(filename.c_str(), x, y, color_size, data, 0);
      break;
    case hash_compile_time("bmp"):
      return_ = stbi_write_bmp(filename.c_str(), x, y, color_size, data);
      break;
    case hash_compile_time("tga"):
      return_ = stbi_write_tga(filename.c_str(), x, y, color_size, data);
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
  Picture(const Index &i) { pic.reShape(i); }
  Picture(const Picture &p) : pic(p.pic) {}
  Picture(const Picture &&p) : pic(p.pic) {}
  ~Picture() {}

  inline int getChannel() const { return color_size; }
  inline void imread(std::string filename) { imread_(filename); }
  inline int imwrite(std::string filename) { return imwrite_(filename); }
  inline bool show(const std::string &colTabStr = "|   ") {
    return pic.show(colTabStr);
  }
  Index shape() {
    Index i_ = pic.shape();
    i_.push_back(color_size);
    return i_;
  }

  void reShape(const Index &shape) { pic.reShape(shape); }

  Tuples<ValueType, color_size> &operator[](const Index &index_) {
    return pic[index_];
  }

  friend Picture operator+(Picture first, Picture second) {
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

  friend Picture operator-(Picture first, Picture second) {
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

  void for_each(std::function<void(Tuples<ValueType, color_size> &)> func) {
    std::for_each(pic.begin(), pic.end(), func);
  }

  void text() {
    // std::cout << pic.end() << std::endl;
  }
};
} // namespace dym
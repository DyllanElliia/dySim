#pragma once

#include "./dyMath.hpp"

template <typename Type, int color_size> struct Pixel {
  using ValueType = Type;
  // #define color_size 3
  std::vector<ValueType> color;
  Pixel(std::vector<ValueType> color_) {
    color.assign(color_.begin(), color_.begin() + color_size);
  }
  Pixel() { color.reserve(color_size); }
  Pixel(ValueType v) { color.resize(color_size, v); }
  Pixel(Pixel &p) : color(p.color) {}
  Pixel(Pixel &&p) : color(p.color) {}

  inline int size() const { return color_size; }

  template <class tranFun, class T>
  static T computer(const T &first, const T &second) {
    T result;
    std::transform(first.color.begin(), first.color.end(), second.color.begin(),
                   std::back_inserter(result.color), tranFun());
    return result;
  }

  void operator=(const Pixel &p) { color = p.color; }
  void operator=(const ValueType &v) {
    for (auto &i : color)
      i = v;
  }

  friend std::ostream &operator<<(std::ostream &output, Pixel &p) {
    // std::cout << p.color[0] << p.color[1] << p.color[2] << std::endl;
    output << "(";
    for (int i = 0; i < p.size(); ++i)
      output << " " << p.color[i];
    output << " )";
    return output;
  }

  friend Pixel operator+(const Pixel &first, const Pixel &second) {
    return computer<std::plus<ValueType>>(first, second);
  }
  friend Pixel operator+(const int &first, Pixel &second) {
    Pixel<ValueType, color_size> result(second);
    for (auto &i : result)
      i = first + i;
    return result;
  }
  friend Pixel operator+(Pixel &first, const int &second) {
    Pixel<ValueType, color_size> result(first);
    for (auto &i : result)
      i = i + second;
    return result;
  }

  friend Pixel operator-(const Pixel &first, const Pixel &second) {
    return computer<std::minus<ValueType>>(first, second);
  }
  friend Pixel operator-(const int &first, Pixel &second) {
    Pixel<ValueType, color_size> result(second);
    for (auto &i : result)
      i = first - i;
    return result;
  }
  friend Pixel operator-(Pixel &first, const int &second) {
    Pixel<ValueType, color_size> result(first);
    for (auto &i : result)
      i = i - second;
    return result;
  }

  friend Pixel operator*(const Pixel &first, const Pixel &second) {
    return computer<std::multiplies<ValueType>>(first, second);
  }
  friend Pixel operator*(const int &first, Pixel &second) {
    Pixel<ValueType, color_size> result(second);
    for (auto &i : result)
      i = first * i;
    return result;
  }
  friend Pixel operator*(Pixel &first, const int &second) {
    Pixel<ValueType, color_size> result(first);
    for (auto &i : result)
      i = i * second;
    return result;
  }

  friend Pixel operator/(const Pixel &first, const Pixel &second) {
    return computer<std::divides<ValueType>>(first, second);
  }
  friend Pixel operator/(const int &first, Pixel &second) {
    Pixel<ValueType, color_size> result(second);
    for (auto &i : result)
      i = first / i;
    return result;
  }
  friend Pixel operator/(Pixel &first, const int &second) {
    Pixel<ValueType, color_size> result(first);
    for (auto &i : result)
      i = i / second;
    return result;
  }

  ValueType &operator[](const int &index_) {
    return color[index_ % color_size];
  }

  void text() { std::cout << (int)color.begin() << std::endl; }
};

#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"

#include "./tools/str_hash.h"

template <typename T = short, int color_size = 3> class Picture {
public:
  using ValueType = T;
  Matrix<Pixel<ValueType, color_size>> pic;
  int channel;

  void imread_(std::string filename) {
    int x, y;
    std::cout << "here3" << std::endl;
    unsigned char *data =
        stbi_load(filename.c_str(), &x, &y, &channel, color_size);
    std::cout << "here" << std::endl;
    try {
      if (channel > color_size)
        throw "imread warning: Picture's color_size must be larger than "
              "picture's channel!";
      std::cout << "heret" << std::endl;
      if (data == nullptr)
        throw "imread error: Image reading failure.";
    } catch (const char *str) {
      std::cerr << str << '\n';
      exit(EXIT_FAILURE);
    }
    std::cout << "heref" << std::endl;
    pic.reShape(gi(x, y));
    std::cout << pic.text() << std::endl;
    std::cout << "herer" << std::endl;
    int yc = y * channel;
    for (int j = 0; j < y; ++j)
      for (int i = 0; i < x; ++i)
        for (int k = 0; k < color_size; ++k) {
          // std::cout << (ValueType)data[i * yc + j * color_size + k]
          //           << std::endl;
          pic[gi(i, j)][k] = (ValueType)data[i * yc + j * color_size + k];
          // std::cout << pic[gi(i, j)][k] << std::endl;
        }
    std::cout << "herei" << std::endl;
    stbi_image_free(data);
  }

  int imwrite_(std::string filename) {
    Index size_ = pic.shape();
    int &x = size_[0], &y = size_[1];
    int yc = y * channel, size_i = x * yc;
    std::cout << "here " << size_i << std::endl;
    unsigned char *data = new unsigned char[size_i];

    for (int j = 0; j < y; ++j)
      for (int i = 0; i < x; ++i)
        for (int k = 0; k < color_size; ++k) {
          // std::cout << i << " " << j << " " << k << " " << pic[gi(i, j)][k]
          //           << " " << i * yc + j * color_size + k << std::endl;
          data[i * yc + j * color_size + k] = pic[gi(i, j)][k];
        }
    std::cout << "here" << std::endl;
    int fne_r = filename.rfind('.');
    if (fne_r == std::string::npos) {
      std::cout << "imwrite error: failed to write picture to " << filename
                << std::endl;
      std::cout << "\twithout picture's type e.g. ~/pic.jpg" << std::endl;
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
    default:
      std::cout << "imwrite error: dyPicture does not support write " +
                       filename_end + "."
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

  void text() {
    // std::cout << pic.end() << std::endl;
  }
};

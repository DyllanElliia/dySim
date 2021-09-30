/*
 * @Author: DyllanElliia
 * @Date: 2021-09-24 15:03:37
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-09-27 16:50:53
 * @Description:
 */

#pragma once
#include "./Index.hpp"

template <typename Type, int color_size> struct Pixel {
  using ValueType = Type;
  // #define color_size 3
  std::vector<ValueType> color;
  Pixel(std::vector<ValueType> color_) {
    color.assign(color_.begin(), color_.begin() + color_size);
  }
  Pixel() { color.resize(color_size, 0); }
  Pixel(ValueType v) { color.resize(color_size, v); }
  Pixel(Pixel &p) : color(p.color) {}
  Pixel(Pixel &&p) : color(p.color) {}

  inline int size() const {
    // qprint(color_size);
    return color_size;
  }

  // Pixel computer(const Pixel &first, const Pixel &second,
  //                std::function<ValueType(ValueType, ValueType)> const
  //                tranFun) {
  //   Pixel<ValueType, color_size>result;
  //   for (int i = 0; i<color_size; ++i) {
  //     result[i] = tranFun(first[i], second[i]);
  //   }
  //   // std::transform(first.color.begin(), first.color.end(),
  //   // second.color.begin(),
  //   //                std::back_inserter(result.color), tranFun());
  //   std::cout << first << second << result << std::endl;
  //   system("pause");
  //   // exit(EXIT_SUCCESS);
  //   return result;
  // }

  void operator=(const Pixel &p) { color = p.color; }
  void operator=(const ValueType &v) {
    for (auto &i : color)
      i = v;
  }
  void operator+=(const Pixel &p) {
    for (size_t i = 0; i < color.size(); ++i)
      color[i] += p.color[i];
  }
  void operator+=(const ValueType &v) {
    for (auto &i : color)
      i += v;
  }

  friend std::ostream &operator<<(std::ostream &output, const Pixel &p) {
    // std::cout << p.color[0] << p.color[1] << p.color[2] << std::endl;
    std::string r = "(";
    for (int i = 0; i < p.size(); ++i)
      r += " " + std::to_string(p.color[i]);
    output << r + " )";
    // std::cout << r + ")" << std::endl;
    return output;
  }

  friend Pixel operator+(const Pixel &first, const Pixel &second) {
    Pixel<ValueType, color_size> result;
    auto &r_ = result.color;
    const auto &f_ = first.color, &s_ = second.color;
    for (int i = 0; i < color_size; ++i) {
      r_[i] = f_[i] + s_[i];
    }
    return result;
  }
  friend Pixel operator+(const ValueType &first, Pixel &second) {
    Pixel<ValueType, color_size> result(second);
    for (auto &i : result.color)
      i = first + i;
    return result;
  }
  friend Pixel operator+(Pixel &first, const ValueType &second) {
    Pixel<ValueType, color_size> result(first);
    for (auto &i : result.color)
      i = i + second;
    return result;
  }

  friend Pixel operator-(const Pixel &first, const Pixel &second) {
    // std::cout << 111 << std::endl;
    // std::cout << first << second << std::endl;
    // system("pause");
    Pixel<ValueType, color_size> result;
    auto &r_ = result.color;
    const auto &f_ = first.color, &s_ = second.color;
    for (int i = 0; i < color_size; ++i) {
      r_[i] = f_[i] - s_[i];
    }
    return result;
  }
  friend Pixel operator-(const ValueType &first, Pixel &second) {
    Pixel<ValueType, color_size> result(second);
    // std::cout << "\t1\t" << result.color[0] << std::endl;
    for (auto &i : result.color)
      i = first - i;
    return result;
  }
  friend Pixel operator-(Pixel &first, const ValueType &second) {
    Pixel<ValueType, color_size> result(first);
    // std::cout << "\t1\t" << result.color[0] << std::endl;
    for (auto &i : result.color)
      i = i - second;
    // std::cout << "\t2\t" << result.color[0] << std::endl;
    return result;
  }

  friend Pixel operator*(const Pixel &first, const Pixel &second) {
    Pixel<ValueType, color_size> result;
    auto &r_ = result.color;
    const auto &f_ = first.color, &s_ = second.color;
    for (int i = 0; i < color_size; ++i) {
      r_[i] = f_[i] * s_[i];
    }
    return result;
  }
  friend Pixel operator*(const ValueType &first, Pixel &second) {
    Pixel<ValueType, color_size> result(second);
    for (auto &i : result.color)
      i = first * i;
    return result;
  }
  friend Pixel operator*(Pixel &first, const ValueType &second) {
    Pixel<ValueType, color_size> result(first);
    for (auto &i : result.color)
      i = i * second;
    return result;
  }

  friend Pixel operator/(const Pixel &first, const Pixel &second) {
    Pixel<ValueType, color_size> result;
    auto &r_ = result.color;
    const auto &f_ = first.color, &s_ = second.color;
    for (int i = 0; i < color_size; ++i) {
      r_[i] = f_[i] / s_[i];
    }
    return result;
  }
  friend Pixel operator/(const ValueType &first, Pixel &second) {
    Pixel<ValueType, color_size> result(second);
    for (auto &i : result.color)
      i = first / i;
    return result;
  }
  friend Pixel operator/(Pixel &first, const ValueType &second) {
    Pixel<ValueType, color_size> result(first);
    for (auto &i : result.color)
      i = i / second;
    return result;
  }

  ValueType &operator[](const int &index_) {
    return color[index_ % color_size];
  }

  void show() {
    std::cout << "(";
    for (int i = 0; i < color_size; ++i)
      std::cout << " " << color[i];
    std::cout << " )\n";
  }

  void text() {}
};

// template <class ValueType, int color_size>
// using Pixel = std::array<ValueType, color_size>;

// template <class ValueType, int color_size>
// std::ostream &operator<<(std::ostream &output,
//                          Pixel<ValueType, color_size> &p) {
//   // std::cout << p[0] << p[1] << p[2] << std::endl;
//   output << "(";
//   for (int i = 0; i < color_size; ++i)
//     output << " " << p[i];
//   output << " )";
//   return output;
// }

// template <class ValueType, int color_size>
// Pixel<ValueType, color_size>
// operator+(const Pixel<ValueType, color_size> &first,
//           const Pixel<ValueType, color_size> &second) {
//   Pixel<ValueType, color_size> result = first;
//   int sn = 0;
//   for (auto &i : result.color)
//     i = i + second[sn++];
//   return result;
// }
// template <class ValueType, int color_size>
// Pixel<ValueType, color_size> operator+(const int &first,
//                                        Pixel<ValueType, color_size> &second)
//                                        {
//   Pixel<ValueType, color_size> result = second;
//   for (auto &i : result.color)
//     i = first + i;
//   return result;
// }
// template <class ValueType, int color_size>
// Pixel<ValueType, color_size> operator+(Pixel<ValueType, color_size> &first,
//                                        const int &second) {
//   Pixel<ValueType, color_size> result = first;
//   for (auto &i : result.color)
//     i = i + second;
//   return result;
// }

// template <class ValueType, int color_size>
// Pixel<ValueType, color_size>
// operator-(const Pixel<ValueType, color_size> &first,
//           const Pixel<ValueType, color_size> &second) {
//   Pixel<ValueType, color_size> result = first;
//   int sn = 0;
//   for (auto &i : result.color)
//     i = i - second[sn++];
//   return result;
// }

// template <class ValueType, int color_size>
// Pixel<ValueType, color_size> operator-(const int &first,
//                                        Pixel<ValueType, color_size> &second)
//                                        {
//   Pixel<ValueType, color_size> result = second;
//   for (auto &i : result.color)
//     i = first - i;
//   return result;
// }
// template <class ValueType, int color_size>
// Pixel<ValueType, color_size> operator-(Pixel<ValueType, color_size> &first,
//                                        const int &second) {
//   Pixel<ValueType, color_size> result = first;
//   for (auto &i : result.color)
//     i = i - second;
//   return result;
// }

// template <class ValueType, int color_size>
// Pixel<ValueType, color_size>
// operator*(const Pixel<ValueType, color_size> &first,
//           const Pixel<ValueType, color_size> &second) {
//   Pixel<ValueType, color_size> result = first;
//   int sn = 0;
//   for (auto &i : result.color)
//     i = i * second[sn++];
//   return result;
// }
// template <class ValueType, int color_size>
// Pixel<ValueType, color_size> operator*(const int &first,
//                                        Pixel<ValueType, color_size> &second)
//                                        {
//   Pixel<ValueType, color_size> result = second;
//   for (auto &i : result.color)
//     i = first * i;
//   return result;
// }
// template <class ValueType, int color_size>
// Pixel<ValueType, color_size> operator*(Pixel<ValueType, color_size> &first,
//                                        const int &second) {
//   Pixel<ValueType, color_size> result = first;
//   for (auto &i : result.color)
//     i = i * second;
//   return result;
// }

// template <class ValueType, int color_size>
// Pixel<ValueType, color_size>
// operator/(const Pixel<ValueType, color_size> &first,
//           const Pixel<ValueType, color_size> &second) {
//   Pixel<ValueType, color_size> result = first;
//   int sn = 0;
//   for (auto &i : result.color)
//     i = i / second[sn++];
//   return result;
// }
// template <class ValueType, int color_size>
// Pixel<ValueType, color_size> operator/(const int &first,
//                                        Pixel<ValueType, color_size> &second)
//                                        {
//   Pixel<ValueType, color_size> result = second;
//   for (auto &i : result.color)
//     i = first / i;
//   return result;
// }
// template <class ValueType, int color_size>
// Pixel<ValueType, color_size> operator/(Pixel<ValueType, color_size> &first,
//                                        const int &second) {
//   Pixel<ValueType, color_size> result = first;
//   for (auto &i : result.color)
//     i = i / second;
//   return result;
// }
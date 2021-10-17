/*
 * @Author: DyllanElliia
 * @Date: 2021-09-24 15:03:37
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-10-17 16:51:46
 * @Description:
 */

#pragma once
#include "./Index.hpp"
namespace dym {
template <typename Type, int color_size> struct Tuples {
  using ValueType = Type;
  // #define color_size 3
  std::vector<ValueType> color;
  Tuples(std::vector<ValueType> color_) {
    color.assign(color_.begin(), color_.begin() + color_size);
  }
  Tuples() { color.resize(color_size, 0); }
  Tuples(ValueType v) { color.resize(color_size, v); }
  Tuples(const Tuples &p) : color(p.color) {}
  Tuples(const Tuples &&p) : color(p.color) {}

  inline int size() const {
    // qprint(color_size);
    return color_size;
  }

  // Tuples computer(const Tuples &first, const Tuples &second,
  //                std::function<ValueType(ValueType, ValueType)> const
  //                tranFun) {
  //   Tuples<ValueType, color_size>result;
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

  void operator=(const Tuples &p) { color = p.color; }
  void operator=(const ValueType &v) {
    for (auto &i : color)
      i = v;
  }
  void operator+=(const Tuples &p) {
    for (size_t i = 0; i < color.size(); ++i)
      color[i] += p.color[i];
  }
  void operator+=(const ValueType &v) {
    for (auto &i : color)
      i += v;
  }

  friend std::ostream &operator<<(std::ostream &output, const Tuples &p) {
    // std::cout << p.color[0] << p.color[1] << p.color[2] << std::endl;
    std::string r = "(";
    for (int i = 0; i < p.size(); ++i)
      r += " " + std::to_string(p.color[i]);
    output << r + " )";
    // std::cout << r + ")" << std::endl;
    return output;
  }

  friend Tuples operator+(const Tuples &first, const Tuples &second) {
    Tuples<ValueType, color_size> result;
    auto &r_ = result.color;
    const auto &f_ = first.color, &s_ = second.color;
    for (int i = 0; i < color_size; ++i) {
      r_[i] = f_[i] + s_[i];
    }
    return result;
  }
  friend Tuples operator+(const ValueType &first, Tuples &second) {
    Tuples<ValueType, color_size> result(second);
    for (auto &i : result.color)
      i = first + i;
    return result;
  }
  friend Tuples operator+(Tuples &first, const ValueType &second) {
    Tuples<ValueType, color_size> result(first);
    for (auto &i : result.color)
      i = i + second;
    return result;
  }

  friend Tuples operator-(const Tuples &first, const Tuples &second) {
    // std::cout << 111 << std::endl;
    // std::cout << first << second << std::endl;
    // system("pause");
    Tuples<ValueType, color_size> result;
    auto &r_ = result.color;
    const auto &f_ = first.color, &s_ = second.color;
    for (int i = 0; i < color_size; ++i) {
      r_[i] = f_[i] - s_[i];
    }
    return result;
  }
  friend Tuples operator-(const ValueType &first, Tuples &second) {
    Tuples<ValueType, color_size> result(second);
    // std::cout << "\t1\t" << result.color[0] << std::endl;
    for (auto &i : result.color)
      i = first - i;
    return result;
  }
  friend Tuples operator-(Tuples &first, const ValueType &second) {
    Tuples<ValueType, color_size> result(first);
    // std::cout << "\t1\t" << result.color[0] << std::endl;
    for (auto &i : result.color)
      i = i - second;
    // std::cout << "\t2\t" << result.color[0] << std::endl;
    return result;
  }

  friend Tuples operator*(const Tuples &first, const Tuples &second) {
    Tuples<ValueType, color_size> result;
    auto &r_ = result.color;
    const auto &f_ = first.color, &s_ = second.color;
    for (int i = 0; i < color_size; ++i) {
      r_[i] = f_[i] * s_[i];
    }
    return result;
  }
  friend Tuples operator*(const ValueType &first, Tuples &second) {
    Tuples<ValueType, color_size> result(second);
    for (auto &i : result.color)
      i = first * i;
    return result;
  }
  friend Tuples operator*(Tuples &first, const ValueType &second) {
    Tuples<ValueType, color_size> result(first);
    for (auto &i : result.color)
      i = i * second;
    return result;
  }

  friend Tuples operator/(const Tuples &first, const Tuples &second) {
    Tuples<ValueType, color_size> result;
    auto &r_ = result.color;
    const auto &f_ = first.color, &s_ = second.color;
    for (int i = 0; i < color_size; ++i) {
      r_[i] = f_[i] / s_[i];
    }
    return result;
  }
  friend Tuples operator/(const ValueType &first, Tuples &second) {
    Tuples<ValueType, color_size> result(second);
    for (auto &i : result.color)
      i = first / i;
    return result;
  }
  friend Tuples operator/(Tuples &first, const ValueType &second) {
    Tuples<ValueType, color_size> result(first);
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
// using Tuples = std::array<ValueType, color_size>;

// template <class ValueType, int color_size>
// std::ostream &operator<<(std::ostream &output,
//                          Tuples<ValueType, color_size> &p) {
//   // std::cout << p[0] << p[1] << p[2] << std::endl;
//   output << "(";
//   for (int i = 0; i < color_size; ++i)
//     output << " " << p[i];
//   output << " )";
//   return output;
// }

// template <class ValueType, int color_size>
// Tuples<ValueType, color_size>
// operator+(const Tuples<ValueType, color_size> &first,
//           const Tuples<ValueType, color_size> &second) {
//   Tuples<ValueType, color_size> result = first;
//   int sn = 0;
//   for (auto &i : result.color)
//     i = i + second[sn++];
//   return result;
// }
// template <class ValueType, int color_size>
// Tuples<ValueType, color_size> operator+(const int &first,
//                                        Tuples<ValueType, color_size> &second)
//                                        {
//   Tuples<ValueType, color_size> result = second;
//   for (auto &i : result.color)
//     i = first + i;
//   return result;
// }
// template <class ValueType, int color_size>
// Tuples<ValueType, color_size> operator+(Tuples<ValueType, color_size> &first,
//                                        const int &second) {
//   Tuples<ValueType, color_size> result = first;
//   for (auto &i : result.color)
//     i = i + second;
//   return result;
// }

// template <class ValueType, int color_size>
// Tuples<ValueType, color_size>
// operator-(const Tuples<ValueType, color_size> &first,
//           const Tuples<ValueType, color_size> &second) {
//   Tuples<ValueType, color_size> result = first;
//   int sn = 0;
//   for (auto &i : result.color)
//     i = i - second[sn++];
//   return result;
// }

// template <class ValueType, int color_size>
// Tuples<ValueType, color_size> operator-(const int &first,
//                                        Tuples<ValueType, color_size> &second)
//                                        {
//   Tuples<ValueType, color_size> result = second;
//   for (auto &i : result.color)
//     i = first - i;
//   return result;
// }
// template <class ValueType, int color_size>
// Tuples<ValueType, color_size> operator-(Tuples<ValueType, color_size> &first,
//                                        const int &second) {
//   Tuples<ValueType, color_size> result = first;
//   for (auto &i : result.color)
//     i = i - second;
//   return result;
// }

// template <class ValueType, int color_size>
// Tuples<ValueType, color_size>
// operator*(const Tuples<ValueType, color_size> &first,
//           const Tuples<ValueType, color_size> &second) {
//   Tuples<ValueType, color_size> result = first;
//   int sn = 0;
//   for (auto &i : result.color)
//     i = i * second[sn++];
//   return result;
// }
// template <class ValueType, int color_size>
// Tuples<ValueType, color_size> operator*(const int &first,
//                                        Tuples<ValueType, color_size> &second)
//                                        {
//   Tuples<ValueType, color_size> result = second;
//   for (auto &i : result.color)
//     i = first * i;
//   return result;
// }
// template <class ValueType, int color_size>
// Tuples<ValueType, color_size> operator*(Tuples<ValueType, color_size> &first,
//                                        const int &second) {
//   Tuples<ValueType, color_size> result = first;
//   for (auto &i : result.color)
//     i = i * second;
//   return result;
// }

// template <class ValueType, int color_size>
// Tuples<ValueType, color_size>
// operator/(const Tuples<ValueType, color_size> &first,
//           const Tuples<ValueType, color_size> &second) {
//   Tuples<ValueType, color_size> result = first;
//   int sn = 0;
//   for (auto &i : result.color)
//     i = i / second[sn++];
//   return result;
// }
// template <class ValueType, int color_size>
// Tuples<ValueType, color_size> operator/(const int &first,
//                                        Tuples<ValueType, color_size> &second)
//                                        {
//   Tuples<ValueType, color_size> result = second;
//   for (auto &i : result.color)
//     i = first / i;
//   return result;
// }
// template <class ValueType, int color_size>
// Tuples<ValueType, color_size> operator/(Tuples<ValueType, color_size> &first,
//                                        const int &second) {
//   Tuples<ValueType, color_size> result = first;
//   for (auto &i : result.color)
//     i = i / second;
//   return result;
// }
}
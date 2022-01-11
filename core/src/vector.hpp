/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 14:32:58
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-07 12:45:35
 * @Description:
 */
#pragma once
#include "define.hpp"
namespace dym {
template <typename Type, int rank>
struct Vector {
 private:
  std::array<Type, rank> a;

 public:
  Vector(const Type &num = 0) {
    for (auto &i : a) i = num;
  }
  template <typename... args>
  Vector(Type v1, args... v) {
    a = {v1, (Type(v))...};
  }
  Vector(std::function<void(Type &)> fun) {
    for (auto &e : a) fun(e);
  }
  Vector(std::function<void(Type &, int)> fun) {
    int i = 0;
    for (auto &e : a) fun(e, i++);
  }
  template <int inRank>
  Vector(const Vector<Type, inRank> &v, const Type &vul = 0) {
    std::memcpy(
        a.data(), v.data(),
        std::min(sizeof(Vector<Type, inRank>), sizeof(Vector<Type, rank>)));
    for (int i = inRank; i < rank; ++i) a[i] = vul;
  }
  Vector(const Vector<Type, rank> &&v) {
    std::memcpy(a.data(), v.a.data(), sizeof(Vector));
  }
  Vector(const Vector<Type, rank> &v) {
    std::memcpy(a.data(), v.a.data(), sizeof(Vector));
  }

  void show() const {
    std::string res = "Vec: [";
    for (auto &i : a) res += std::to_string(i) + " ";
    res += "]";
    std::cout << res << std::endl;
  }
  inline auto data() const { return a.data(); }

  Type &operator[](const int &i) { return a[i]; }
  Type operator[](const int &i) const { return a[i]; }
  Vector operator=(const Vector &v) {
    memcpy(a, v.a, sizeof(Vector));
    return *this;
  }
  Vector operator=(const Type &num) {
    for (auto &i : a) i = num;
    return *this;
  }
  friend std::ostream &operator<<(std::ostream &output, const Vector &v) {
    std::string res = "Vec: [";
    for (auto &i : v.a) res += std::to_string(i) + " ";
    res += "]\n";
    output << res;
    return output;
  }
};
#define _dym_vector_operator_binary_(op)                                       \
  template <typename Type, int rank>                                           \
  Vector<Type, rank> operator op(const Vector<Type, rank> &f,                  \
                                 const Vector<Type, rank> &s) {                \
    return Vector<Type, rank>([&](Type &e, int i) { e = f[i] op s[i]; });      \
  }                                                                            \
  template <typename Type, int rank>                                           \
  Vector<Type, rank> operator op(const Type &f, const Vector<Type, rank> &s) { \
    return Vector<Type, rank>([&](Type &e, int i) { e = f op s[i]; });         \
  }                                                                            \
  template <typename Type, int rank>                                           \
  Vector<Type, rank> operator op(const Vector<Type, rank> &f, const Type &s) { \
    return Vector<Type, rank>([&](Type &e, int i) { e = f[i] op s; });         \
  }

#define _dym_vector_operator_unary_(op)                    \
  template <typename Type, int rank>                       \
  void operator op(Vector<Type, rank> &f, const Type &s) { \
    for (int i = 0; i < rank; ++i) f[i] op s;              \
  }
_dym_vector_operator_binary_(+) _dym_vector_operator_binary_(-)
    _dym_vector_operator_binary_(*) _dym_vector_operator_binary_(/)
        _dym_vector_operator_unary_(+=) _dym_vector_operator_unary_(-=)
            _dym_vector_operator_unary_(*=) _dym_vector_operator_unary_(/=)

}  // namespace dym

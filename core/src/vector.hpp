/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 14:32:58
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-12 16:46:04
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
  // template <Type... args>
  Vector(std::array<Type, rank> v) { a = v; }
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
  template <int inRank>
  Vector operator=(const Vector<Type, inRank> &v) {
    memcpy(a.data(), v.data(),
           std::min(sizeof(Vector<Type, inRank>), sizeof(Vector<Type, rank>)));
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
  template <typename cType>
  Vector<cType, rank> cast() {
    Vector<cType, rank> o;
    for (int i = 0; i < rank; ++i) o[i] = a[i];
    return o;
  }
  Type dot(const Vector &v) const {
    Type res = 0;
    for (int i = 0; i < rank; ++i) res += a[i] * v[i];
    return res;
  }
};

template <typename Type, int rank>
constexpr Type dot(const Vector<Type, rank> &v1, const Vector<Type, rank> &v2) {
  return v1.dot(v2);
}

template <typename Type, int rank>
constexpr Type operator*(const Vector<Type, rank> &f,
                         const Vector<Type, rank> &s) {
  return dot(f, s);
}

template <typename Type, int rank>
Vector<Type, rank> operator*(const Type &f, const Vector<Type, rank> &s) {
  return Vector<Type, rank>([&](Type &e, int i) { e = f * s[i]; });
}
template <typename Type, int rank>
Vector<Type, rank> operator*(const Vector<Type, rank> &f, const Type &s) {
  return Vector<Type, rank>([&](Type &e, int i) { e = f[i] * s; });
}

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
_dym_vector_operator_binary_(+);
_dym_vector_operator_binary_(-);
// _dym_vector_operator_binary_(*);
_dym_vector_operator_binary_(/);
_dym_vector_operator_unary_(+=);
_dym_vector_operator_unary_(-=);
_dym_vector_operator_unary_(*=);
_dym_vector_operator_unary_(/=);

}  // namespace dym

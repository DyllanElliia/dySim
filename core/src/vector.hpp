/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 14:32:58
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-20 16:25:25
 * @Description:
 */
#pragma once
#include "Index.hpp"
namespace dym {
template <typename Type, std::size_t dim>
struct Vector {
 private:
  std::array<Type, dim> a;

 public:
  Vector(const Type &num = 0) {
    for (auto &i : a) i = num;
  }
  // template <Type... args>
  Vector(std::array<Type, dim> v) { a = v; }
  Vector(std::function<void(Type &)> fun) {
    for (auto &e : a) fun(e);
  }
  Vector(std::function<void(Type &, int)> fun) {
    int i = 0;
    for (auto &e : a) fun(e, i++);
  }
  template <std::size_t inDim>
  Vector(const Vector<Type, inDim> &v, const Type &vul = 0) {
    std::memcpy(
        a.data(), v.data(),
        std::min(sizeof(Vector<Type, inDim>), sizeof(Vector<Type, dim>)));
    for (int i = inDim; i < dim; ++i) a[i] = vul;
  }
  Vector(const Vector<Type, dim> &&v) {
    std::memcpy(a.data(), v.a.data(), sizeof(Vector));
  }
  Vector(const Vector<Type, dim> &v) {
    std::memcpy(a.data(), v.a.data(), sizeof(Vector));
  }

  void show() const {
    std::string res = "Vec: [";
    for (auto &i : a) res += std::to_string(i) + " ";
    res += "]";
    std::cout << res << std::endl;
  }
  inline auto data() const { return a.data(); }
  inline void for_each(std::function<void(Type &)> func) {
    for (auto &e : a) func(e);
  }
  inline void for_each(std::function<void(Type &, int)> func) {
    int i = 0;
    for (auto &e : a) func(e, i++);
  }

  Type &operator[](const int &i) { return a[i]; }
  Type operator[](const int &i) const { return a[i]; }

#define _dym_vector_xyzw_(whichOne, index)                         \
  inline Type whichOne() const {                                   \
    if constexpr (dim > index)                                     \
      return a[index];                                             \
    else {                                                         \
      printf("Error: Only Vector's dim>=1 can use whichOne()!\n"); \
      return a[index];                                             \
    }                                                              \
  }
  _dym_vector_xyzw_(x, 0);
  _dym_vector_xyzw_(y, 1);
  _dym_vector_xyzw_(z, 2);
  _dym_vector_xyzw_(w, 3);

  template <std::size_t inDim>
  inline Vector operator=(const Vector<Type, inDim> &v) {
    memcpy(a.data(), v.data(),
           std::min(sizeof(Vector<Type, inDim>), sizeof(Vector<Type, dim>)));
    return *this;
  }
  inline Vector operator=(const Vector &v) {
    memcpy(a.data(), v.a.data(), sizeof(Vector));
    return *this;
  }
  inline Vector operator=(const Type &num) {
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
  inline Vector<cType, dim> cast() {
    return Vector<cType, dim>([&](cType &e, int i) { e = a[i]; });
  }
  Type dot(const Vector &v) const {
    Type res = 0;
    for (int i = 0; i < dim; ++i) res += a[i] * v[i];
    return res;
  }
};

template <typename Type, std::size_t dim>
inline Type dot(const Vector<Type, dim> &v1, const Vector<Type, dim> &v2) {
  return v1.dot(v2);
}

template <typename Type, std::size_t dim>
inline Type operator*(const Vector<Type, dim> &f, const Vector<Type, dim> &s) {
  return dot(f, s);
}

template <typename Type, std::size_t dim>
inline Vector<Type, dim> operator-(const Vector<Type, dim> &v) {
  return Vector<Type, dim>([&](Type &e, int i) { e = -v[i]; });
}

#define _dym_vector_type_operator_binary_(op)                         \
  template <typename Type, std::size_t dim>                           \
  inline Vector<Type, dim> operator op(const Type &f,                 \
                                       const Vector<Type, dim> &s) {  \
    return Vector<Type, dim>([&](Type &e, int i) { e = f op s[i]; }); \
  }                                                                   \
  template <typename Type, std::size_t dim>                           \
  inline Vector<Type, dim> operator op(const Vector<Type, dim> &f,    \
                                       const Type &s) {               \
    return Vector<Type, dim>([&](Type &e, int i) { e = f[i] op s; }); \
  }

_dym_vector_type_operator_binary_(*);
_dym_vector_type_operator_binary_(/);

#define _dym_vector_operator_binary_(op)                                 \
  template <typename Type, std::size_t dim>                              \
  inline Vector<Type, dim> operator op(const Vector<Type, dim> &f,       \
                                       const Vector<Type, dim> &s) {     \
    return Vector<Type, dim>([&](Type &e, int i) { e = f[i] op s[i]; }); \
  }                                                                      \
  _dym_vector_type_operator_binary_(op);

#define _dym_vector_operator_unary_(op)                          \
  template <typename Type, std::size_t dim>                      \
  inline void operator op(Vector<Type, dim> &f, const Type &s) { \
    for (int i = 0; i < dim; ++i) f[i] op s;                     \
  }

_dym_vector_operator_binary_(+);
_dym_vector_operator_binary_(-);
// _dym_vector_operator_binary_(*);
// _dym_vector_operator_binary_(/);
_dym_vector_operator_unary_(+=);
_dym_vector_operator_unary_(-=);
_dym_vector_operator_unary_(*=);
_dym_vector_operator_unary_(/=);

}  // namespace dym

/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 14:32:58
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-25 16:28:11
 * @Description:
 */
#pragma once
#include "Index.hpp"
#include "realALG.hpp"
#include "launch.hpp"
namespace dym {
template <typename Type, std::size_t dim>
struct Vector {
 private:
  std::array<Type, dim> a;

 public:
  Vector(const Type &num = 0) {
    Loop<int, dim>([&](auto i) { a[i] = num; });
    // for (auto &i : a) i = num;
  }
  // template <Type... args>
  Vector(const std::array<Type, dim> &v) { a = v; }
  Vector(std::function<void(Type &)> fun) {
    Loop<int, dim>([&](auto i) { fun(a[i]); });
    // for (auto &e : a) fun(e);
  }
  Vector(std::function<void(Type &, int)> fun) {
    Loop<int, dim>([&](auto i) { fun(a[i], i); });
    // int i = 0;
    // for (auto &e : a) fun(e, i++);
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
  constexpr _DYM_FORCE_INLINE_ auto data() const { return a.data(); }
  _DYM_FORCE_INLINE_ void for_each(std::function<void(Type &)> func) {
    // for (auto &e : a) func(e);
    Loop<int, dim>([&](auto i) { func(a[i]); });
  }
  _DYM_FORCE_INLINE_ void for_each(std::function<void(Type &, int)> func) {
    // int i = 0;
    // for (auto &e : a) func(e, i++);
    Loop<int, dim>([&](auto i) { func(a[i], i); });
  }

  Type &operator[](const int &i) { return a[i]; }
  Type operator[](const int &i) const { return a[i]; }

#define _dym_vector_xyzw_(whichOne, index)                         \
  _DYM_FORCE_INLINE_ Type whichOne() const {                       \
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
    res += "]";
    output << res;
    return output;
  }
  template <typename cType>
  inline Vector<cType, dim> cast() const {
    return Vector<cType, dim>([&](cType &e, int i) { e = a[i]; });
  }
  inline Type dot(const Vector &v) const {
    Type res = 0;
    Loop<int, dim>([&](auto i) { res += a[i] * v[i]; });
    // for (int i = 0; i < dim; ++i) res += a[i] * v[i];
    return res;
  }
  template <typename... Vs>
  inline Vector<Type, dim> cross(Vs... vec) const;

  inline Type length_sqr() const {
    Type ans = 0;
    Loop<int, dim>([&](auto i) { ans += dym::sqr(a[i]); });
    return ans;
  }
  constexpr _DYM_FORCE_INLINE_ Type length() const {
    return dym::sqrt(length_sqr());
  }

  constexpr _DYM_FORCE_INLINE_ Vector<Type, dim> normalize() const {
    return *this / length();
  }

  _DYM_FORCE_INLINE_ Vector<Type, dim> reflect(
      const Vector<Type, dim> &normal) const {
    auto &vec = *this;
    return vec - 2 * (vec.dot(normal)) * normal;
  }

  constexpr _DYM_FORCE_INLINE_ auto shape() const { return dim; }

};

namespace vector {
template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Type dot(const Vector<Type, dim> &v1,
                            const Vector<Type, dim> &v2) {
  return v1.dot(v2);
}

template <typename Type, std::size_t dim, typename... Vs>
_DYM_FORCE_INLINE_ Vector<Type, dim> cross(const Vector<Type, dim> &v,
                                           Vs... vec) {
  return v.cross(vec...);
}

template <typename Type, std::size_t dim>
constexpr _DYM_FORCE_INLINE_ Type normalized(const Vector<Type, dim> &v) {
  return v.normalize();
}
}  // namespace vector

// template <typename Type, std::size_t dim>
// _DYM_FORCE_INLINE_ Type operator*(const Vector<Type, dim> &f, const
// Vector<Type, dim> &s)
// {
//   return dot(f, s);
// }

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

// _dym_vector_type_operator_binary_(*);
_dym_vector_type_operator_binary_(/);
template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ Vector<Type, dim> operator/(const Vector<Type, dim> &f,
                                               const Vector<Type, dim> &s) {
  return f;
}

#define _dym_vector_operator_binary_(op)                                 \
  template <typename Type, std::size_t dim>                              \
  inline Vector<Type, dim> operator op(const Vector<Type, dim> &f,       \
                                       const Vector<Type, dim> &s) {     \
    return Vector<Type, dim>([&](Type &e, int i) { e = f[i] op s[i]; }); \
  }                                                                      \
  _dym_vector_type_operator_binary_(op);

#define _dym_vector_operator_unary_(op)                                       \
  template <typename Type, std::size_t dim>                                   \
  inline void operator op(Vector<Type, dim> &f, const Vector<Type, dim> &s) { \
    for (int i = 0; i < dim; ++i) f[i] op s[i];                               \
  }                                                                           \
  template <typename Type, std::size_t dim>                                   \
  inline void operator op(Vector<Type, dim> &f, const Type &s) {              \
    for (int i = 0; i < dim; ++i) f[i] op s;                                  \
  }

_dym_vector_operator_binary_(+);
_dym_vector_operator_binary_(-);
_dym_vector_operator_binary_(*);
// _dym_vector_operator_binary_(/);
_dym_vector_operator_unary_(+=);
_dym_vector_operator_unary_(-=);
_dym_vector_operator_unary_(*=);
_dym_vector_operator_unary_(/=);

#define _dym_vector_operator_cmp_unary_(op)                            \
  template <typename Type, std::size_t dim>                            \
  inline bool operator op(const Vector<Type, dim> &f,                  \
                          const Vector<Type, dim> &s) {                \
    for (int i = 0; i < dim; ++i)                                      \
      if (!(f[i] op s[i])) return false;                               \
    return true;                                                       \
  }                                                                    \
  template <typename Type, std::size_t dim>                            \
  inline bool operator op(const Vector<Type, dim> &f, const Type &s) { \
    for (int i = 0; i < dim; ++i)                                      \
      if (!(f[i] op s)) return false;                                  \
    return true;                                                       \
  }

_dym_vector_operator_cmp_unary_(<);
_dym_vector_operator_cmp_unary_(<=);
_dym_vector_operator_cmp_unary_(>);
_dym_vector_operator_cmp_unary_(>=);
// _dym_vector_operator_cmp_unary_(==);

template <typename Type, std::size_t dim>
inline bool operator==(const Vector<Type, dim> &f, const Vector<Type, dim> &s) {
  for (int i = 0; i < dim; ++i)
    if (!(dym::abs(f[i] - s[i]) < 1e-7)) return false;
  return true;
}
template <typename Type, std::size_t dim>
inline bool operator==(const Vector<Type, dim> &f, const Type &s) {
  for (int i = 0; i < dim; ++i)
    if (!(dym::abs(f[i] - s) < 1e-7)) return false;
  return true;
}

}  // namespace dym

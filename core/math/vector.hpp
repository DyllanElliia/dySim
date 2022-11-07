/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 14:32:58
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-19 15:19:54
 * @Description:
 */
#pragma once
#include "Index.hpp"
#include "glm/detail/qualifier.hpp"
#include "glm/matrix.hpp"
#include "launch.hpp"
#include "realALG.hpp"
#include <initializer_list>

namespace dym {
template <typename Type, std::size_t dim> struct Vector {
private:
  std::array<Type, dim> a;

public:
  _DYM_GENERAL_ Vector() {}
  _DYM_GENERAL_ Vector(const Type &num) {
    Loop<int, dim>([&](auto i) { a[i] = num; });
    // for (auto &i : a) i = num;
  }
  // template <Type... args>
  _DYM_GENERAL_ Vector(const std::array<Type, dim> &v) { a = v; }
  _DYM_GENERAL_ Vector(const std::initializer_list<Type> &v) {
    // Loop<int, dim>([&](auto i) { a[i] = v[i]; });
    short i = 0;
    for (auto &obj : v)
      a[i++] = obj;
  }

  //   template <
  //       typename F1,
  //       std::enable_if_t<
  //           std::is_convertible<F1, std::function<void(Type &)>>::value, int>
  //           = 0>
  //   _DYM_GENERAL_ Vector(const F1 &fun) {
  // #ifdef __CUDA_ARCH__
  //     for (int i = 0; i < dim; ++i)
  //       fun(a[i]);
  // #else
  //     Loop<int, dim>([&] (auto i) { fun(a[i]); });
  // #endif
  //   }

  DYM_TEMPLATE_CHECK(F, std::function<void(Type &, int)>)
  Vector(const F &fun) {
    Loop<int, dim>([&](auto i) { fun(a[i], i); });
  }
  DYM_TEMPLATE_CHECK(F, std::function<Type(int)>)
  _DYM_GENERAL_ Vector(const F &funf) {
    Loop<int, dim>([&](auto i) { a[i] = funf(i); });
  }
  template <std::size_t inDim>
  _DYM_GENERAL_ Vector(const Vector<Type, inDim> &v, const Type &vul = 0) {
    std::memcpy(
        a.data(), v.data(),
        std::min(sizeof(Vector<Type, inDim>), sizeof(Vector<Type, dim>)));
    for (int i = inDim; i < dim; ++i)
      a[i] = vul;
  }
  _DYM_GENERAL_ Vector(const Vector<Type, dim> &&v) {
    std::memcpy(a.data(), v.a.data(), sizeof(Vector));
  }
  _DYM_GENERAL_ Vector(const Vector<Type, dim> &v) {
    std::memcpy(a.data(), v.a.data(), sizeof(Vector));
  }

  template <typename glmmT = float, glm::qualifier glmtp = glm::defaultp>
  _DYM_GENERAL_ Vector(const glm::vec<dim, glmmT, glmtp> &&v) {
    Loop<int, dim>([&](auto i) { a[i] = v[i]; });
  }
  template <typename glmmT = float, glm::qualifier glmtp = glm::defaultp>
  _DYM_GENERAL_ Vector(const glm::vec<dim, glmmT, glmtp> &v) {
    Loop<int, dim>([&](auto i) { a[i] = v[i]; });
  }

  void show() const { std::cout << *this << std::endl; }
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

  _DYM_GENERAL_ Type &operator[](const int &i) { return a[i]; }
  _DYM_GENERAL_ Type operator[](const int &i) const { return a[i]; }

#define _dym_vector_xyzw_(whichOne, index)                                     \
  _DYM_GENERAL_ _DYM_FORCE_INLINE_ Type whichOne() const {                     \
    if constexpr (dim > index)                                                 \
      return a[index];                                                         \
    else {                                                                     \
      printf("Error: Only Vector's dim>=1 can use whichOne()!\n");             \
      return a[index];                                                         \
    }                                                                          \
  }
  _dym_vector_xyzw_(x, 0);
  _dym_vector_xyzw_(y, 1);
  _dym_vector_xyzw_(z, 2);
  _dym_vector_xyzw_(w, 3);

  template <std::size_t inDim>
  _DYM_GENERAL_ inline Vector operator=(const Vector<Type, inDim> &v) {
    memcpy(a.data(), v.data(),
           std::min(sizeof(Vector<Type, inDim>), sizeof(Vector<Type, dim>)));
    return *this;
  }
  _DYM_GENERAL_ inline Vector operator=(const Vector &v) {
    memcpy(a.data(), v.a.data(), sizeof(Vector));
    return *this;
  }
  template <typename glmmT = float, glm::qualifier glmtp = glm::defaultp>
  _DYM_GENERAL_ inline Vector operator=(const glm::vec<dim, glmmT, glmtp> &v) {
    Loop<int, dim>([&](auto i) { a[i] = v[i]; });
    return *this;
  }
  _DYM_GENERAL_ inline Vector operator=(const Type &num) {
    for (auto &i : a)
      i = num;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &output, const Vector &v) {
    output << "Vec: [";
    for (auto &i : v.a)
      output << i << " ";
    output << "]";
    return output;
  }
  template <typename cType>
  _DYM_GENERAL_ inline Vector<cType, dim> cast() const {
    return Vector<cType, dim>([=](int i) { return (cType)a[i]; });
  }
  _DYM_GENERAL_ inline Type dot(const Vector &v) const {
    Type res = 0;
    Loop<int, dim>([&](auto i) { res += a[i] * v[i]; });
    // for (int i = 0; i < dim; ++i) res += a[i] * v[i];
    return res;
  }
  template <typename... Vs>
  _DYM_GENERAL_ inline Vector<Type, dim> cross(Vs... vec) const;

  _DYM_GENERAL_ inline Type length_sqr() const {
    Type ans = 0;
    Loop<int, dim>([&](auto i) { ans += dym::sqr(a[i]); });
    return ans;
  }
  _DYM_GENERAL_ constexpr _DYM_FORCE_INLINE_ Type length() const {
    return dym::sqrt(length_sqr());
  }

  _DYM_GENERAL_ constexpr _DYM_FORCE_INLINE_ Vector<Type, dim>
  normalize() const {
    return *this / length();
  }

  _DYM_GENERAL_ _DYM_FORCE_INLINE_ Vector<Type, dim>
  reflect(const Vector<Type, dim> &normal) const {
    auto &vec = *this;
    return vec - 2 * (vec.dot(normal)) * normal;
  }

  _DYM_GENERAL_ _DYM_FORCE_INLINE_ auto transpose() const;

  _DYM_GENERAL_ constexpr _DYM_FORCE_INLINE_ auto shape() const { return dim; }

  template <typename glmmT = float, glm::qualifier glmtp = glm::defaultp>
  _DYM_GENERAL_ _DYM_FORCE_INLINE_ glm::vec<dim, glmmT, glmtp>
  to_glm_vec() const {
    glm::vec<dim, glmmT, glm::defaultp> res;
    Loop<int, dim>([&](auto i) { res[i] = a[i]; });
    return res;
  }

  template <typename glmmT = float, glm::qualifier glmtp = glm::defaultp>
  _DYM_GENERAL_ operator glm::vec<dim, glmmT, glmtp>() const {
    return to_glm_vec();
  }
};

namespace vector {
template <typename Type, std::size_t dim>
_DYM_GENERAL_ _DYM_FORCE_INLINE_ Type dot(const Vector<Type, dim> &v1,
                                          const Vector<Type, dim> &v2) {
  return v1.dot(v2);
}

template <typename Type, std::size_t dim, typename... Vs>
_DYM_GENERAL_ _DYM_FORCE_INLINE_ Vector<Type, dim>
cross(const Vector<Type, dim> &v, Vs... vec) {
  return v.cross(vec...);
}

template <typename Type, std::size_t dim>
_DYM_GENERAL_ constexpr _DYM_FORCE_INLINE_ Type
normalized(const Vector<Type, dim> &v) {
  return v.normalize();
}
} // namespace vector

// template <typename Type, std::size_t dim>
// _DYM_FORCE_INLINE_ Type operator*(const Vector<Type, dim> &f, const
// Vector<Type, dim> &s)
// {
//   return dot(f, s);
// }

template <typename Type, std::size_t dim>
_DYM_GENERAL_ inline Vector<Type, dim> operator-(const Vector<Type, dim> &v) {
  return Vector<Type, dim>([=](int i) { return -v[i]; });
}

#define _dym_vector_type_operator_binary_(op)                                  \
  template <typename Type, typename TypeS, std::size_t dim>                    \
  _DYM_GENERAL_ inline Vector<Type, dim> operator op(                          \
      const TypeS &f, const Vector<Type, dim> &s) {                            \
    return Vector<Type, dim>([=](int i) { return f op s[i]; });                \
  }                                                                            \
  template <typename Type, typename TypeS, std::size_t dim>                    \
  _DYM_GENERAL_ inline Vector<Type, dim> operator op(                          \
      const Vector<Type, dim> &f, const TypeS &s) {                            \
    return Vector<Type, dim>([=](int i) { return f[i] op s; });                \
  }

// _dym_vector_type_operator_binary_(*);
_dym_vector_type_operator_binary_(/);
template <typename Type, std::size_t dim>
_DYM_GENERAL_ _DYM_FORCE_INLINE_ Vector<Type, dim>
operator/(const Vector<Type, dim> &f, const Vector<Type, dim> &s) {
  return Vector<Type, dim>([=](int i) { return f[i] / s[i]; });
}

#define _dym_vector_operator_binary_(op)                                       \
  template <typename Type, std::size_t dim>                                    \
  _DYM_GENERAL_ inline Vector<Type, dim> operator op(                          \
      const Vector<Type, dim> &f, const Vector<Type, dim> &s) {                \
    return Vector<Type, dim>([=](int i) { return f[i] op s[i]; });             \
  }                                                                            \
  _dym_vector_type_operator_binary_(op);

#define _dym_vector_operator_unary_(op)                                        \
  template <typename Type, std::size_t dim>                                    \
  _DYM_GENERAL_ inline void operator op(Vector<Type, dim> &f,                  \
                                        const Vector<Type, dim> &s) {          \
    for (int i = 0; i < dim; ++i)                                              \
      f[i] op s[i];                                                            \
  }                                                                            \
  template <typename Type, typename TypeS, std::size_t dim>                    \
  _DYM_GENERAL_ inline void operator op(Vector<Type, dim> &f,                  \
                                        const TypeS &s) {                      \
    for (int i = 0; i < dim; ++i)                                              \
      f[i] op s;                                                               \
  }

_dym_vector_operator_binary_(+);
_dym_vector_operator_binary_(-);
_dym_vector_operator_binary_(*);
// _dym_vector_operator_binary_(/);
_dym_vector_operator_unary_(+=);
_dym_vector_operator_unary_(-=);
_dym_vector_operator_unary_(*=);
_dym_vector_operator_unary_(/=);

#define _dym_vector_operator_cmp_unary_(op)                                    \
  template <typename Type, std::size_t dim>                                    \
  _DYM_GENERAL_ inline bool operator op(const Vector<Type, dim> &f,            \
                                        const Vector<Type, dim> &s) {          \
    for (int i = 0; i < dim; ++i)                                              \
      if (!(f[i] op s[i]))                                                     \
        return false;                                                          \
    return true;                                                               \
  }                                                                            \
  template <typename Type, typename TypeS, std::size_t dim>                    \
  _DYM_GENERAL_ inline bool operator op(const Vector<Type, dim> &f,            \
                                        const TypeS &s) {                      \
    for (int i = 0; i < dim; ++i)                                              \
      if (!(f[i] op s))                                                        \
        return false;                                                          \
    return true;                                                               \
  }

_dym_vector_operator_cmp_unary_(<);
_dym_vector_operator_cmp_unary_(<=);
_dym_vector_operator_cmp_unary_(>);
_dym_vector_operator_cmp_unary_(>=);
// _dym_vector_operator_cmp_unary_(==);

template <typename Type, std::size_t dim>
_DYM_GENERAL_ inline bool operator==(const Vector<Type, dim> &f,
                                     const Vector<Type, dim> &s) {
  for (int i = 0; i < dim; ++i)
    if (!(dym::abs(f[i] - s[i]) < 1e-7))
      return false;
  return true;
}
template <typename Type, typename TypeS, std::size_t dim>
_DYM_GENERAL_ inline bool operator==(const Vector<Type, dim> &f,
                                     const TypeS &s) {
  for (int i = 0; i < dim; ++i)
    if (!(dym::abs(f[i] - s) < 1e-7))
      return false;
  return true;
}

} // namespace dym

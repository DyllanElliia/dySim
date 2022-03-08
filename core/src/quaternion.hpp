/*
 * @Author: DyllanElliia
 * @Date: 2022-03-08 15:16:29
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-08 16:56:14
 * @Description:
 */
#pragma once
#include "matALG.hpp"
namespace dym {
template <typename Type = Real>
class Quaternion {
 public:
  Type w;
  Vector<Type, 3> v;

  Quaternion(const Type& vul) : w(vul), v(vul) {}
  Quaternion(const Type& w, const Type& x, const Type& y, const Type& z)
      : w(w), v({x, y, z}) {}
  Quaternion(const Type& w, const Vector<Type, 3>& v) : w(w), v(v) {}
  Quaternion(const Quaternion& q) : w(q.w), v(q.v) {}
  Quaternion(const Quaternion&& q) : w(q.w), v(q.v) {}

  inline Quaternion operator=(const Quaternion& q) {
    w = q.w, v = q.v;
    return *this;
  }
  inline Quaternion operator=(const Type& vul) {
    w = vul, v = vul;
    return *this;
  }

  _DYM_FORCE_INLINE_ Type norm_sqr() const { return w * w + v.length_sqr(); }
  _DYM_FORCE_INLINE_ Type norm() const { return sqrt(norm_sqr()); }

  _DYM_FORCE_INLINE_ Quaternion normalize() const { return *this / norm(); }

  _DYM_FORCE_INLINE_ Quaternion conjugate() const {return Quaternion(w, -v)}

  _DYM_FORCE_INLINE_ Quaternion inverse() const {
    return conjugate() / norm_sqr();
  }
  _DYM_FORCE_INLINE_ Quaternion inv() const { return inverse(); }

  _DYM_FORCE_INLINE_ Matrix<Type, 3, 3> to_matrix() const {
    const auto &a = w, &b = v[0], &c = v[1], &d = v[2];
    return Matrix<Type, 3, 3>(
        {{1 - 2 * (sqr(c) + sqr(d)), 2 * (b * c - a * d), 2 * (a * c + b * d)},
         {2 * (b * c + a * d), 1 - 2 * (sqr(b) + sqr(d)), 2 * (c * d - a * b)},
         {2 * (b * d - a * c), 2 * (a * b + c * d),
          1 - 2 * (sqr(b) + sqr(c))}});
  }

  _DYM_FORCE_INLINE_ Quaternion pow(const Type& t) {
    const Type theta = acos(w);
    const Vector<Type, 3> u = v / sin(theta);
    return Quaternion(cos(t * theta), sin(t * theta) * u);
  }

  _DYM_FORCE_INLINE_ Type dot(const Quaternion& q) {
    return w * q.w + v.dot(q.v);
  }

  _DYM_FORCE_INLINE_ Quaternion cross(const Quaternion& q) { return *this * q; }
};

#define _dym_quaternion_type_operator_binary_(op)                  \
  template <typename Type, typename TypeS>                         \
  inline Quaternion<Type> operator op(const TypeS& f,              \
                                      const Quaternion<Type>& s) { \
    return Quaternion<Type>(f op s.w, f op s.v);                   \
  }                                                                \
  template <typename Type, typename TypeS>                         \
  inline Quaternion<Type> operator op(const Quaternion<Type>& f,   \
                                      const TypeS& s) {            \
    return Quaternion<Type>(f op s.w, f op s.v);                   \
  }

#define _dym_quaternion_type_operator_unary_(op)                 \
  template <typename Type, typename TypeS>                       \
  inline void operator op(Quaternion<Type>& f, const TypeS& s) { \
    f.w op s, f.v op s;                                          \
  }

_dym_quaternion_type_operator_binary_(+);
_dym_quaternion_type_operator_binary_(-);
_dym_quaternion_type_operator_binary_(*);
_dym_quaternion_type_operator_binary_(/);
_dym_quaternion_type_operator_unary_(+=);
_dym_quaternion_type_operator_unary_(-=);
_dym_quaternion_type_operator_unary_(*=);
_dym_quaternion_type_operator_unary_(/=);

#define _dym_quaternion_operator_binary_(op)                       \
  template <typename Type>                                         \
  inline Quaternion<Type> operator op(const Quaternion<Type>& f,   \
                                      const Quaternion<Type>& s) { \
    return Quaternion<Type>(f.w op s.w, f.v op s.v);               \
  }

#define _dym_quaternion_operator_unary_(op)                                 \
  template <typename Type>                                                  \
  inline void operator op(Quaternion<Type>& f, const Quaternion<Type>& s) { \
    f.w op s.w, f.v op s.v;                                                 \
  }

_dym_quaternion_operator_binary_(+);
_dym_quaternion_operator_binary_(-);
_dym_quaternion_operator_unary_(+=);
_dym_quaternion_operator_unary_(-=);

// GraBmann Product
template <typename Type>
inline Quaternion<Type> operator*(const Quaternion<Type>& q1,
                                  const Quaternion<Type>& q2) {
  // q1=[a, v(bi,cj,dk)], q2=[e, u(fi,gj,hk)]
  const auto &a = q1.w, &e = q2.w;
  const auto &v = q1.v, &u = q2.v;
  return Quaternion<Type>(a * e - v.dot(u), a * u + e * v + v.cross(u));
}

// GraBmann Product
template <typename Type>
inline void operator*=(Quaternion<Type>& q1, const Quaternion<Type>& q2) {
  // q1=[a, v(bi,cj,dk)], q2=[e, u(fi,gj,hk)]
  auto& a = q1.w;
  const auto& e = q2.w;
  auto& v = q1.v;
  const auto& u = q2.v;
  q1.w = a * e - v.dot(u), q1.v = a * u + e * v + v.cross(u);
}

template <typename Type = Real>
_DYM_FORCE_INLINE_ Quaternion<Type> getQuaternion(const Type& theta,
                                                  const Vector<Type, 3>& u) {
  return Quaternion<Type>(cos(0.5 * theta, sin(0.5 * theta) * u));
}
namespace quaternion {
template <typename Type>
_DYM_FORCE_INLINE_ Quaternion<Type> lerp(const Quaternion<Type>& q0,
                                         const Quaternion<Type>& q1,
                                         const Real& t) {
  return (1 - t) * q0 + t * q1;
}
template <typename Type>
_DYM_FORCE_INLINE_ Quaternion<Type> Nlerp(const Quaternion<Type>& q0,
                                          const Quaternion<Type>& q1,
                                          const Real& t) {
  Quaternion<Type> r = lerp(q0, q1, t);
  return r.normalize();
}
template <typename Type>
_DYM_FORCE_INLINE_ Quaternion<Type> Slerp(const Quaternion<Type>& q0,
                                          const Quaternion<Type>& q1,
                                          const Real& t) {
  Type theta = 1 / cos(q0.dot(q1)), sintheta_inv = 1 / sin(theta);
  return sin((1 - t) * theta) * sintheta_inv * q0 +
         sin(t * theta) * sintheta_inv * q1;
}
}  // namespace quaternion

}  // namespace dym
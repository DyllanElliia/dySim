/*
 * @Author: DyllanElliia
 * @Date: 2022-07-12 15:10:35
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-13 16:46:18
 * @Description:
 */
#pragma once
#include "./dual_num.hpp"

namespace dym {

namespace AD {
namespace {
template <typename... Args> struct dx_s { std::tuple<Args...> arg; };
template <typename... Args> struct var_s { std::tuple<Args...> arg; };

template <typename A, typename B>
constexpr _DYM_FORCE_INLINE_ auto is_same_pkg(const A &a, const B &b) {
  return std::is_same<std::decay<A>, std::decay<B>>::value;
}

template <typename Type> _DYM_FORCE_INLINE_ Type unpkg_dual(const Type &v) {
  return Type(0);
};
template <typename Type>
_DYM_FORCE_INLINE_ Type unpkg_dual(const Dual<Type> &v) {
  return v.B;
};
template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ auto unpkg_dual(const Vector<Dual<Type>, dim> &v) {
  return Vector<Type, dim>([&](Type &obj, int i) { obj = v[i].B; });
};

template <typename Type, std::size_t m, std::size_t n>
_DYM_FORCE_INLINE_ auto unpkg_dual(const Matrix<Dual<Type>, m, n> &v) {
  return Matrix<Type, m, n>([&](Type &obj, int i, int j) { obj = v[i][j].B; });
}

template <typename Type>
constexpr _DYM_FORCE_INLINE_ void seed(Type &a, const Real &vul) {}
template <typename Type>
constexpr _DYM_FORCE_INLINE_ void seed(Dual<Type> &a, const Real &vul) {
  a.B = vul;
}
template <typename Type, std::size_t dim>
constexpr _DYM_FORCE_INLINE_ void seed(Vector<Dual<Type>, dim> &a,
                                       const Real &vul) {
  a.for_each([&](Dual<Type> &ai) { ai.B = vul; });
}
} // namespace

template <typename... Args> auto fc(Args &...args) {
  return dx_s<Args &...>{std::forward_as_tuple(std::forward<Args>(args)...)};
}

template <typename... Args> auto all(Args &...args) {
  return var_s<Args &...>{std::forward_as_tuple(std::forward<Args>(args)...)};
}

template <typename Func, typename dVar, typename... Vars>
auto dx(const Func &function, const Dual<dVar> &dx_,
        const var_s<Vars...> &variable) {
  constexpr const auto VarSize = sizeof...(Vars);
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    if (is_same_pkg(dx_, vag) && (void *)&dx_ == (void *)&vag)
      seed(vag, 1.0);
    else
      seed(vag, 0.0);
  });
  auto u = std::apply(function, variable.arg);
  return unpkg_dual(u);
}

template <typename Func, typename dVar, std::size_t dim, typename... Vars>
auto dx(const Func &function, const Vector<Dual<dVar>, dim> &dx_,
        const var_s<Vars...> &variable) {
  constexpr const auto VarSize = sizeof...(Vars);
  Vector<decltype(std::apply(function, variable.arg)), dim> res(0.0);
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    seed(vag, 0.0);
  });
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    if (is_same_pkg(dx_, vag) && (void *)&dx_ == (void *)&vag) {
      Loop<int, dim>([&](auto ii) {
        seed(vag[ii], 1.0);
        res[ii] = std::apply(function, variable.arg);
        seed(vag[ii], 0.0);
      });
    }
  });
  return unpkg_dual(res);
}

template <typename Func, typename dVar, std::size_t m, std::size_t n,
          typename... Vars>
auto dx(const Func &function, const Matrix<Dual<dVar>, m, n> &dx_,
        const var_s<Vars...> &variable) {
  constexpr const auto VarSize = sizeof...(Vars);
  Matrix<decltype(std::apply(function, variable.arg)), m, n> res(0.0);
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    seed(vag, 0.0);
  });
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    if (is_same_pkg(dx_, vag) && (void *)&dx_ == (void *)&vag) {
      Loop<int, m>([&](auto int ii) {
        Loop<int, n>([&](auto jj) {
          seed(vag[ii][jj], 1.0);
          res[ii][jj] = unpkg_dual(std::apply(function, variable.arg));
          seed(vag[ii][jj], 0.0);
        });
      });
    }
  });
  return unpkg_dual(res);
}

template <typename Type, typename funT, typename... dVars, typename... Vars>
auto d(const funT &function, const dx_s<dVars...> &dx_,
       const var_s<Vars...> &variable) {}
} // namespace AD
} // namespace dym
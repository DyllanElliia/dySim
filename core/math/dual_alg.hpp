/*
 * @Author: DyllanElliia
 * @Date: 2022-07-12 15:10:35
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-12 16:56:01
 * @Description:
 */
#pragma once
#include "./dual_num.hpp"
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace dym {

namespace AD {
namespace {
template <typename... Args> struct dx_s { std::tuple<Args...> arg; };
template <typename... Args> struct var_s { std::tuple<Args...> arg; };

template <typename A, typename B>
constexpr _DYM_FORCE_INLINE_ auto is_same_pkg(const A &a, const B &b) {
  return std::is_same<std::decay<A>, std::decay<B>>::value;
}
template <typename Type>
constexpr _DYM_FORCE_INLINE_ void seed(Type &a, const Real &vul) {}
template <typename Type>
constexpr _DYM_FORCE_INLINE_ void seed(Dual<Type> &a, const Real &vul) {
  a.B = vul;
}
} // namespace

template <typename... Args> auto fc(Args &...args) {
  return dx_s<Args &...>{std::forward_as_tuple(std::forward<Args>(args)...)};
}

template <typename... Args> auto all(Args &...args) {
  return var_s<Args &...>{std::forward_as_tuple(std::forward<Args>(args)...)};
}

template <typename funT, typename dVar, typename... Vars>
auto dx(const funT &function, const dVar &dx_, const var_s<Vars...> &variable) {

  constexpr const auto VarSize = sizeof...(Vars);
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    if (is_same_pkg(dx_, vag) && (void *)&dx_ == (void *)&vag)
      seed(vag, 1.0);
    else
      seed(vag, 0.0);
    qprint(vag);
  });
  auto u = std::apply(function, variable.arg);
  return u.B;
}

template <typename Type, typename funT, typename... dVars, typename... Vars>
auto d(const funT &function, const dx_s<dVars...> &dx_,
       const var_s<Vars...> &variable) {}
} // namespace AD
} // namespace dym
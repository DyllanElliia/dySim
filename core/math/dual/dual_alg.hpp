/*
 * @Author: DyllanElliia
 * @Date: 2022-07-12 15:10:35
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-19 16:23:37
 * @Description:
 */
#pragma once
#include "./dual_num.hpp"
#include <tuple>
#include <type_traits>

namespace dym {

namespace AD {
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

template <typename Type, std::size_t dim, std::size_t dimt>
_DYM_FORCE_INLINE_ auto
unpkg_dual(const Vector<Vector<Dual<Type>, dimt>, dim> &v) {
  return Vector<Vector<Type, dimt>, dim>(
      [&](Vector<Type, dimt> &obj, int i) { obj = unpkg_dual(v[i]); });
};

template <typename Type, std::size_t m, std::size_t n, std::size_t dimt>
_DYM_FORCE_INLINE_ auto
unpkg_dual(const Matrix<Vector<Dual<Type>, dimt>, m, n> &v) {
  return Matrix<Vector<Type, dimt>, m, n>(
      [&](Vector<Type, dimt> &obj, int i, int j) {
        obj = unpkg_dual(v[i][j]);
      });
}

template <typename Type, std::size_t dim, std::size_t mt, std::size_t nt>
_DYM_FORCE_INLINE_ auto
unpkg_dual(const Vector<Matrix<Dual<Type>, mt, nt>, dim> &v) {
  return Vector<Matrix<Type, mt, nt>, dim>(
      [&](Matrix<Type, mt, nt> &obj, int i) { obj = unpkg_dual(v[i]); });
};

template <typename Type, std::size_t m, std::size_t n, std::size_t mt,
          std::size_t nt>
_DYM_FORCE_INLINE_ auto
unpkg_dual(const Matrix<Matrix<Dual<Type>, mt, nt>, m, n> &v) {
  return Matrix<Matrix<Type, mt, nt>, m, n>(
      [&](Matrix<Type, mt, nt> &obj, int i, int j) {
        obj = unpkg_dual(v[i][j]);
      });
}

template <typename Type> _DYM_FORCE_INLINE_ Type pkg_dual(const Type &v) {
  return Dual<Type>(v);
};
template <typename Type, std::size_t dim>
_DYM_FORCE_INLINE_ auto pkg_dual(const Vector<Type, dim> &v) {
  return Vector<Dual<Type>, dim>([&](Type &obj, int i) { obj = v[i].B; });
};

template <typename Type, std::size_t m, std::size_t n>
_DYM_FORCE_INLINE_ auto pkg_dual(const Matrix<Type, m, n> &v) {
  return Matrix<Dual<Type>, m, n>(
      [&](Type &obj, int i, int j) { obj = v[i][j].B; });
}
namespace {
template <typename... Args> struct dx_s { std::tuple<Args...> arg; };
template <typename... Args> struct var_s { std::tuple<Args...> arg; };

template <typename A, typename B, typename funT1, typename funT2>
auto is_same_pkg(const A &a, const B &b, funT1 funcTrue, funT2 funcFalse) {
  if constexpr (std::is_same<std::decay<A>, std::decay<B>>::value)
    funcTrue();
  else
    funcFalse();
}

template <typename Type>
constexpr _DYM_FORCE_INLINE_ void seed(Type &a, const Real &vul,
                                       const int &i = -1, const int &j = -1) {}
template <typename Type>
constexpr _DYM_FORCE_INLINE_ void seed(Dual<Type> &a, const Real &vul,
                                       const int &i = -1, const int &j = -1) {
  a.B = vul;
}
template <typename Type, std::size_t dim>
constexpr _DYM_FORCE_INLINE_ void seed(Vector<Dual<Type>, dim> &a,
                                       const Real &vul, const int &i = -1,
                                       const int &j = -1) {
  if (i == -1)
    a.for_each([&](Dual<Type> &ai) { ai.B = vul; });
  else
    a[i].B = vul;
}
template <typename Type, std::size_t m, std::size_t n>
constexpr _DYM_FORCE_INLINE_ void seed(Matrix<Dual<Type>, m, n> &a,
                                       const Real &vul, const int &i = -1,
                                       const int &j = -1) {
  if (i == -1 && j == -1)
    a.for_each([&](Dual<Type> &ai) { ai.B = vul; });
  else
    a[i][j].B = vul;
}

template <typename Type> _DYM_FORCE_INLINE_ Type simplify_vm(const Type &v) {
  return v;
}

template <typename Type, std::size_t dim1, std::size_t dim2>
_DYM_FORCE_INLINE_ Vector<Type, dim1 * dim2>
simplify_vm(const Vector<Vector<Type, dim2>, dim1> &v) {
  return Vector<Type, dim1 * dim2>(
      [&](Type &obj, int i) { obj = v[i / dim2][i % dim2]; });
}

template <typename Type, std::size_t m, std::size_t n, std::size_t dim>
_DYM_FORCE_INLINE_ Matrix<Type, dim * m, n>
simplify_vm(const Vector<Matrix<Type, m, n>, dim> &v) {
  return Matrix<Type, dim * m, n>(
      [&](Type &obj, int i, int j) { obj = v[i / m][i % m][j]; });
}

template <typename Type, std::size_t m, std::size_t n, std::size_t dim>
_DYM_FORCE_INLINE_ Matrix<Type, dim * m, n>
simplify_vm(const Matrix<Vector<Type, dim>, m, n> &v) {
  return Matrix<Type, dim * m, n>(
      [&](Type &obj, int i, int j) { obj = v[i / dim][j][i % dim]; });
}

template <typename Type, std::size_t m1, std::size_t n1, std::size_t m2,
          std::size_t n2>
_DYM_FORCE_INLINE_ Matrix<Type, m1 * m2, n1 * n2>
simplify_vm(const Matrix<Matrix<Type, m2, n2>, m1, n1> &v) {
  return Matrix<Type, m1 * m2, n1 * n2>([&](Type &obj, int i, int j) {
    obj = v[i / m2][j / n2][i % m2][j % n2];
  });
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

    if ((void *)&dx_ == (void *)&vag)
      is_same_pkg(
          dx_, vag, [&]() { seed(vag, 1.0); }, [&]() { seed(vag, 0.0); });
  });
  auto u = std::apply(function, variable.arg);
  return unpkg_dual(u);
}

template <typename Func, typename dVar, std::size_t dim, typename... Vars>
auto dx(const Func &function, const Vector<Dual<dVar>, dim> &dx_,
        const var_s<Vars...> &variable) {
  constexpr const auto VarSize = sizeof...(Vars);
  Vector<decltype(std::apply(function, variable.arg)), dim> res;
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    seed(vag, 0.0);
  });
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    if ((void *)&dx_ == (void *)&vag)
      is_same_pkg(
          dx_, vag,
          [&]() {
            Loop<int, dim>([&](auto ii) {
              seed(vag, 1.0, ii);
              res[ii] = std::apply(function, variable.arg);
              seed(vag, 0.0, ii);
            });
          },
          [&]() {});
  });
  return simplify_vm(unpkg_dual(res));
}

template <typename Func, typename dVar, std::size_t m, std::size_t n,
          typename... Vars>
auto dx(const Func &function, const Matrix<Dual<dVar>, m, n> &dx_,
        const var_s<Vars...> &variable) {
  constexpr const auto VarSize = sizeof...(Vars);
  Matrix<decltype(std::apply(function, variable.arg)), m, n> res;
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    seed(vag, 0.0);
  });
  Loop<int, VarSize>([&](auto i) {
    auto &vag = std::get<i>(variable.arg);
    if ((void *)&dx_ == (void *)&vag)
      is_same_pkg(
          dx_, vag,
          [&]() {
            Loop<int, m>([&](auto ii) {
              Loop<int, n>([&](auto jj) {
                seed(vag, 1.0, ii, jj);
                res[ii][jj] = std::apply(function, variable.arg);
                seed(vag, 0.0, ii, jj);
              });
            });
          },
          [&]() {});
  });
  return simplify_vm(unpkg_dual(res));
}

namespace {
template <typename Type> constexpr Type getDualType(Dual<Type> t) { return t; }

#define getDualType_c(dtype)                                                   \
  template <typename Type>                                                     \
  constexpr auto getDualType(Dual<Type> t, dtype funt) {                       \
    return Type();                                                             \
  }                                                                            \
  template <typename Type, std::size_t dim>                                    \
  constexpr auto getDualType(Dual<Type> t, Vector<dtype, dim> funt) {          \
    return Vector<Type, dim>();                                                \
  }                                                                            \
  template <typename Type, std::size_t m, std::size_t n>                       \
  constexpr auto getDualType(Dual<Type> t, Matrix<dtype, m, n> funt) {         \
    return Matrix<Type, m, n>();                                               \
  }                                                                            \
                                                                               \
  template <typename Type, std::size_t dimt>                                   \
  constexpr auto getDualType(Vector<Dual<Type>, dimt> t, dtype funt) {         \
                                                                               \
    return Vector<Type, dimt>();                                               \
  }                                                                            \
                                                                               \
  template <typename Type, std::size_t dim, std::size_t dimt>                  \
  constexpr auto getDualType(Vector<Dual<Type>, dimt> t,                       \
                             Vector<dtype, dim> funt) {                        \
                                                                               \
    return Vector<Type, dim * dimt>();                                         \
  }                                                                            \
                                                                               \
  template <typename Type, std::size_t m, std::size_t n, std::size_t dim>      \
  constexpr auto getDualType(Vector<Dual<Type>, dim> t,                        \
                             Matrix<dtype, m, n> funt) {                       \
                                                                               \
    return Matrix<Type, m * dim, n>();                                         \
  }                                                                            \
                                                                               \
  template <typename Type, std::size_t mt, std::size_t nt>                     \
  constexpr auto getDualType(Matrix<Dual<Type>, mt, nt> t, dtype funt) {       \
    return Matrix<Type, mt, nt>();                                             \
  }                                                                            \
  template <typename Type, std::size_t dim, std::size_t mt, std::size_t nt>    \
  constexpr auto getDualType(Matrix<Dual<Type>, mt, nt> t,                     \
                             Vector<dtype, dim> funt) {                        \
    return Matrix<Type, mt * dim, nt>();                                       \
  }                                                                            \
  template <typename Type, std::size_t m, std::size_t n, std::size_t mt,       \
            std::size_t nt>                                                    \
  constexpr auto getDualType(Matrix<Dual<Type>, mt, nt> t,                     \
                             Matrix<dtype, m, n> funt) {                       \
    return Matrix<Type, m * mt, n * nt>();                                     \
  }

getDualType_c(Type);
getDualType_c(Dual<Type>);

template <typename Type> constexpr auto rett() {
  std::decay_t<Type> varaaa;
  return varaaa;
}
} // namespace

template <typename funT, typename... dVars, typename... Vars>
auto d(const funT &function, const dx_s<dVars...> &dx_,
       const var_s<Vars...> &variable) {
  std::tuple<decltype(getDualType(rett<dVars>(),
                                  std::apply(function, variable.arg)))...>
      res;
  constexpr const auto VarSize = sizeof...(dVars);
  Loop<int, VarSize>([&](auto i) {
    std::get<i>(res) = dx(function, std::get<i>(dx_.arg), variable);
  });
  return res;
}
} // namespace AD
} // namespace dym
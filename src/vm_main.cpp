/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 15:30:45
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-29 16:51:49
 * @Description:
 */
#include <dyMath.hpp>
#include <random>

#include "../core/math/matALG.hpp"

int main(int argc, char** argv) {
  dym::Vector<Real, 3> a(1);
  dym::Vector<Real, 3> b({1, 2, 3});
  std::cout << a + b << a * b << a - b << b + 1.f << b - 1 << b * 2.f << b / 2
            << b / 1e-4;
  auto c = b;
  c.show();
  c *= 10.f;
  c.show();
  auto d = dym::Vector<Real, 5>(b, 6);
  d.show();
  auto e = dym::Vector<Real, 2>(d);
  e.show();
  dym::Vector<Real, 10> z(10);
  a = z;
  a.show();
  a = e;
  a.show();
  a.cast<int>().show();
  dym::Vector<Real, 10> g([](Real& e, int i) { e = i; });
  g.show();
  g = -g;
  g.show();
  qprint(g.x(), g.y(), g.z(), g.w());
  g = 0;
  g.show();

  qprint("matrix test:");
  dym::Matrix<Real, 3, 4> ma(1);
  auto mb = dym::Matrix<Real, 2, 5>(ma, 10);
  auto mc = dym::Matrix<Real, 4, 3>(ma, 10);
  auto mg = dym::Matrix<Real, 8, 8>(ma, 10);

  auto me = mb;
  dym::Matrix<Real, 2, 5> ml([](Real& e, int i, int j) { e = i * 10 + j; });
  ma.show();
  mb.show();
  mc.show();
  mg.show();
  me.show();
  ml.show();
  ml.cast<int>().show();
  dym::Matrix<int, 10, 10>(ml.cast<int>(), -100).show();
  dym::Matrix<int, 2, 3>({{1, 2, 3}, {4, 5, 6}}).show();
  (ml * d).show();
  (dym::Matrix<int, 3, 2>({{1, 3}, {4, 0}, {2, 1}}) *
   dym::Vector<int, 2>({1, 5}))
      .show();

  dym::Matrix<int, 5, 8> foreachT(0);
  foreachT.for_each([&](int& e, int i, int j) {
    e = i * 10 + j;
    // std::cout << i << " " << j << std::endl;
  });
  foreachT.show();

  Real* pml = &(ml[0][0]);
  qprint("ptr test", pml[0], *(pml + 1), pml[5], pml[7]);

  dym::Matrix<Real, 2, 3> mulB({{1, 2, 3}, {4, 5, 6}});
  dym::Matrix<Real, 3, 2> mulA({{1, 3}, {4, 0}, {2, 1}});

  dym::matrix::mul_std(mulA, mulB).show();
  dym::matrix::mul_swap(mulA, mulB).show();
  dym::matrix::mul_fast(mulA, mulB).show();

  dym::Matrix<Real, 3, 3> o;
  o = mulA * mulB * dym::matrix::identity<Real, 3>(0.5f);
  o.show();
  dym::TimeLog t;

  // long long times = 9223372036854775807;
  long long times = 1e3;
  double scale = 1;
  std::default_random_engine re;
  std::uniform_real_distribution<Real> u(0.f, 1.f);

  qprint_nlb("times =", times);
  qprint();
  unsigned long long count = 0;
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<Real, 80, 100> mulC([&](Real& e) { e = u(re); });
    dym::Matrix<Real, 100, 80> mulD([&](Real& e) { e = u(re); });
  }
  t.record(scale);
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<Real, 80, 100> mulC([&](Real& e) { e = u(re); });
    dym::Matrix<Real, 100, 80> mulD([&](Real& e) { e = u(re); });

    auto CD = dym::matrix::mul_std(mulC, mulD);
  }
  t.record(scale);
  // std::cout << o << std::endl;
  qprint();
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<Real, 80, 100> mulC([&](Real& e) { e = u(re); });
    dym::Matrix<Real, 100, 80> mulD([&](Real& e) { e = u(re); });
    auto CD = dym::matrix::mul_swap(mulC, mulD);
  }
  t.record(scale);
  // std::cout << o << std::endl;
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<Real, 80, 100> mulC([&](Real& e) { e = u(re); });
    dym::Matrix<Real, 100, 80> mulD([&](Real& e) { e = u(re); });
    auto CD = dym::matrix::mul_fast(mulC, mulD);
  }
  t.record(scale);

  times = 1e6;
  qprint_nlb("times =", times);
  qprint();
  count = 0;
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> mulC([&](Real& e) { e = u(re); });
    dym::Matrix<Real, 3, 3> mulD([&](Real& e) { e = u(re); });
  }
  t.record(scale);
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> mulC([&](Real& e) { e = u(re); });
    dym::Matrix<Real, 3, 3> mulD([&](Real& e) { e = u(re); });

    auto CD = dym::matrix::mul_std(mulC, mulD);
  }
  t.record(scale);
  // qprint(t.getRecord() - 1.2);
  // std::cout << o << std::endl;
  qprint();
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> mulC([&](Real& e) { e = u(re); });
    dym::Matrix<Real, 3, 3> mulD([&](Real& e) { e = u(re); });
    auto CD = dym::matrix::mul_swap(mulC, mulD);
  }
  t.record(scale);
  // qprint(t.getRecord() - 1.2);
  // std::cout << o << std::endl;
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<Real, 3, 3> mulC([&](Real& e) { e = u(re); });
    dym::Matrix<Real, 3, 3> mulD([&](Real& e) { e = u(re); });
    auto CD = dym::matrix::mul_fast(mulC, mulD);
  }
  t.record(scale);
  // qprint(t.getRecord() - 1.2);
  // std::cout << o << std::endl;
  qprint(a, b);
  a += b;
  qprint(a);

  qprint(dym::pow(dym::Vector<Real, 3>({1.f, 2.f, 3.f}), 1.2f));
  qprint(dym::min(dym::Vector<int, 3>({-2, 0, 2}), dym::Vector<int, 3>(0)));
  qprint(dym::exp(10.f));

  dym::Loop<int, 10>([&](auto i) { qprint("Loop: ", i); });

  auto c4 = dym::vector::cross(dym::Vector<Real, 4>({1.f, 2.f, 3.f, 4.f}),
                               dym::Vector<Real, 4>({-4.f, -2.f, 3.f, 1.f}),
                               dym::Vector<Real, 4>({10.f, -8.f, 6.f, 5.f}));
  qprint(c4, c4.dot(dym::Vector<Real, 4>({1.f, 2.f, 3.f, 4.f})),
         c4.dot(dym::Vector<Real, 4>({-4.f, -2.f, 3.f, 1.f})),
         c4.dot(dym::Vector<Real, 4>({10.f, -8.f, 6.f, 5.f})));
  qprint(dym::vector::cross(dym::Vector<Real, 3>({1.f, 0.f, 0.f}),
                            dym::Vector<Real, 3>({0.f, 1.f, 0.f})));
  qprint(dym::vector::cross(dym::Vector<Real, 2>({0.3f, 0.2f})));

  qprint(dym::matrix::det(dym::Matrix<Real, 3, 3>(
             {{1.f, 2.f, 3.f}, {0.f, 2.f, 3.f}, {0.f, 0.f, 3.f}})),
         dym::matrix::det(dym::Matrix<Real, 5, 5>({{1.f, 2.f, 3.f, 4.f, 5.f},
                                                   {0.f, 2.f, 3.f, 4.f, 5.f},
                                                   {0.f, 0.f, 3.f, 4.f, 5.f},
                                                   {0.f, 0.f, 0.f, 4.f, 5.f},
                                                   {0.f, 0.f, 0.f, 0.f, 5.f}})),
         dym::Matrix<Real, 5, 5>({{1.f, 2.f, 3.f, 4.f, 5.f},
                                  {0.f, 2.f, 3.f, 4.f, 5.f},
                                  {0.f, 0.f, 3.f, 4.f, 5.f},
                                  {0.f, 0.f, 0.f, 4.f, 5.f},
                                  {0.f, 0.f, 0.f, 0.f, 5.f}})
             .det());
  qprint(dym::Matrix<Real, 5, 5>({{1.f, 2.f, 3.f, 4.f, 5.f},
                                  {0.f, 2.f, 3.f, 4.f, 5.f},
                                  {0.f, 0.f, 3.f, 4.f, 5.f},
                                  {0.f, 0.f, 0.f, 4.f, 5.f},
                                  {0.f, 0.f, 0.f, 0.f, 5.f}})
             .sub(1, 1));
  auto mat_t_inv = dym::Matrix<Real, 3, 3>(
      {{2.f, 6.f, 4.f}, {8.f, 4.f, 2.f}, {9.f, 0.f, 2.f}});
  qprint(mat_t_inv * mat_t_inv.inv());
  qprint(dym::Vector<Real, 3>({1.f, 2.f, 3.f})
             .reflect(dym::Vector<Real, 3>({0.f, 1.f, 0.f})),
         dym::Vector<Real, 3>({1.f, 2.f, 3.f}).normalize().length());

  return 0;
}
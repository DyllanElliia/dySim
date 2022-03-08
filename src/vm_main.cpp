/*
 * @Author: DyllanElliia
 * @Date: 2021-11-23 15:30:45
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-25 16:33:11
 * @Description:
 */
#include "../core/src/matALG.hpp"
#include <dyMath.hpp>
#include <random>

int main(int argc, char** argv) {
  dym::Vector<float, 3> a(1);
  dym::Vector<float, 3> b({1, 2, 3});
  std::cout << a + b << a * b << a - b << b + 1.f << b - 1 << b * 2.f << b / 2
            << b / 1e-4;
  auto c = b;
  c.show();
  c *= 10.f;
  c.show();
  auto d = dym::Vector<float, 5>(b, 6);
  d.show();
  auto e = dym::Vector<float, 2>(d);
  e.show();
  dym::Vector<float, 10> z(10);
  a = z;
  a.show();
  a = e;
  a.show();
  a.cast<int>().show();
  dym::Vector<float, 10> g([](float& e, int i) { e = i; });
  g.show();
  g = -g;
  g.show();
  qprint(g.x(), g.y(), g.z(), g.w());
  g = 0;
  g.show();

  qprint("matrix test:");
  dym::Matrix<float, 3, 4> ma(1);
  auto mb = dym::Matrix<float, 2, 5>(ma, 10);
  auto mc = dym::Matrix<float, 4, 3>(ma, 10);
  auto mg = dym::Matrix<float, 8, 8>(ma, 10);

  auto me = mb;
  dym::Matrix<float, 2, 5> ml([](float& e, int i, int j) { e = i * 10 + j; });
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

  float* pml = &(ml[0][0]);
  qprint("ptr test", pml[0], *(pml + 1), pml[5], pml[7]);

  dym::Matrix<float, 2, 3> mulB({{1, 2, 3}, {4, 5, 6}});
  dym::Matrix<float, 3, 2> mulA({{1, 3}, {4, 0}, {2, 1}});

  dym::matrix::mul_std(mulA, mulB).show();
  dym::matrix::mul_swap(mulA, mulB).show();
  dym::matrix::mul_fast(mulA, mulB).show();

  dym::Matrix<float, 3, 3> o;
  o = mulA * mulB * dym::matrix::identity<float, 3>(0.5f);
  o.show();
  dym::TimeLog t;

  // long long times = 9223372036854775807;
  long long times = 1e2;
  double scale = 1;
  std::default_random_engine re;
  std::uniform_real_distribution<float> u(0.f, 1.f);

  qprint_nlb("times =", times);
  qprint();
  unsigned long long count = 0;
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<float, 80, 100> mulC([&](float& e) { e = u(re); });
    dym::Matrix<float, 100, 80> mulD([&](float& e) { e = u(re); });
  }
  t.record(scale);
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<float, 80, 100> mulC([&](float& e) { e = u(re); });
    dym::Matrix<float, 100, 80> mulD([&](float& e) { e = u(re); });

    auto CD = dym::matrix::mul_std(mulC, mulD);
  }
  t.record(scale);
  // std::cout << o << std::endl;
  qprint();
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<float, 80, 100> mulC([&](float& e) { e = u(re); });
    dym::Matrix<float, 100, 80> mulD([&](float& e) { e = u(re); });
    auto CD = dym::matrix::mul_swap(mulC, mulD);
  }
  t.record(scale);
  // std::cout << o << std::endl;
  t.reStart();
  for (long long i = 0; i < times; ++i) {
    dym::Matrix<float, 80, 100> mulC([&](float& e) { e = u(re); });
    dym::Matrix<float, 100, 80> mulD([&](float& e) { e = u(re); });
    auto CD = dym::matrix::mul_fast(mulC, mulD);
  }
  t.record(scale);
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
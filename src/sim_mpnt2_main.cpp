/*
 * @Author: DyllanElliia
 * @Date: 2022-02-16 15:30:48
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-10 17:34:44
 * @Description:
 */
#define DYM_DEFAULT_THREAD 1
#define _dym_test_
#include <dyGraphic.hpp>
#include <dySimulator.hpp>
#include <random>
typedef dym::Vector<Real, 3> Vector3;

std::default_random_engine re;
std::uniform_real_distribution<Real> u(-1.f, 1.f);
void add_object(Vector3 center, int begin, int end, int material_index) {
  qprint(material_index);
  auto &x = dym::mpmt::x;
  for (int i = begin; i < end; ++i) {
    x[i] = Vector3({u(re), u(re), u(re)}) * 0.1f + center;
    dym::mpmt::particles[i].material = material_index;
  }
}
void add_object2(Vector3 center, int begin, int end, int material_index) {
  auto &x = dym::mpmt::x;
  for (int i = begin; i < end; ++i) {
    x[i] = Vector3({u(re), u(re), u(re)}) * 0.2f + center;
    dym::mpmt::particles[i].material = material_index;
  }
}
int main(int argc, char const *argv[]) {
  dym::GUI gui("mpm-test", dym::gi(17, 47, 65));
  dym::mpmt::init(dym::mpmt::MidGrid);
  u_int n3 = 3000;
  dym::Tensor<Vector3> newX(0, dym::gi(n3));
  newX.for_each_i([&](Vector3 &pos) {
    pos = Vector3({u(re), u(re), u(re)}) * 0.1f;
  });
  dym::mpmt::addParticle(newX + Vector3({0.35, 0.43, 0.5}),
                         dym::mpmt::addLiquidMaterial());
  dym::mpmt::addParticle(newX + Vector3({0.50, 0.64, 0.5}),
                         dym::mpmt::addJellyMaterial());
  dym::mpmt::addParticle(newX + Vector3({0.65, 0.84, 0.5}),
                         dym::mpmt::addPlasticMaterial());
  dym::Tensor<dym::Vector<Real, 2>> point(0, dym::gi(dym::mpmt::n_particles));
  auto Tp = [&](dym::Tensor<Vector3> &x) {
    point.for_each_i([&](dym::Vector<Real, 2> &e, int i) {
      auto &pos = x[i];
      Real PI = 3.14159265f;
      Real phi = (28.0 / 180) * PI, theta = (32.0 / 180) * PI;
      Vector3 a = pos / 1.5f - 0.35f;
      Real c = std::cos(phi), s = std::sin(phi), C = std::cos(theta),
            S = std::sin(theta);
      Real x = a.x() * c + a.z() * s, z = a.z() * c - a.x() * s;
      Real u = x, v = a.y() * C + z * S;
      e[0] = 2 * u, e[1] = 2 * v;
    });
  };
  dym::TimeLog time;
  gui.init(500, 500);
  int step = 0;
  gui.update([&]() {
    time.record();
    time.reStart();
    Tp(dym::mpmt::x);
    gui.scatter2D(point, dym::gi(6, 133, 135), 0, 0, n3);
    gui.scatter2D(point, dym::gi(237, 85, 59), 0, n3, 2 * n3);
    gui.scatter2D(point, dym::gi(255, 255, 255), 0, 2 * n3);
    for (int i = 0; i < dym::mpmt::steps; ++i) dym::mpmt::advance(1e-4), ++step;
  });
}

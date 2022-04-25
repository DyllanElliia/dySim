/*
 * @Author: DyllanElliia
 * @Date: 2022-02-16 15:30:48
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-29 16:17:04
 * @Description:
 */
#define DYM_DEFAULT_THREAD 1
#include <dyGraphic.hpp>
#include <dySimulator.hpp>
#include <random>
typedef dym::Vector<Real, 3> Vector3;

int main(int argc, char const *argv[]) {
  dym::MLSMPM<dym::MidGrid, dym::OneSeparateOtherSticky> sim;
  sim.globalForce = Vector3({0.f, -9.8 * 2.f, 0.f});
  std::default_random_engine re;
  std::uniform_real_distribution<Real> u(-1.f, 1.f);
  u_int n3 = 10000;
  dym::Tensor<Vector3> newX(0, dym::gi(n3));

  newX.for_each_i([&](Vector3 &pos) {
    pos = Vector3({u(re), u(re), u(re)}) * 0.1f;
  });

  sim.addParticle(newX + dym::Vector3({0.35, 0.42, 0.5}),
                  sim.addLiquidMaterial());
  sim.addParticle(newX + dym::Vector3({0.50, 0.63, 0.5}),
                  sim.addJellyMaterial());
  sim.addParticle(newX + dym::Vector3({0.65, 0.84, 0.5}),
                  sim.addPlasticMaterial());
  // newX.for_each_i([&](Vector3 &pos) {
  //   pos = Vector3({u(re), u(re), u(re)}) * 0.25f;
  // });

  // sim.addParticle(newX + Vector3(0.5), sim.addLiquidMaterial());

  dym::GUI gui("simMpm-test", dym::gi(17, 47, 65));
  gui.init(800, 800);
  const Real dt = 1e-4;
  qprint("asdf:", dym::Vector<Real, 4>(dt * sim.globalForce));
  getchar();
  dym::Tensor<dym::Vector<Real, 2>> point(0, dym::gi(sim.getParticlesNum()));
  const int steps = 25;
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
  qprint(sim.getParticlesNum(), n3);
  gui.update([&]() {
    time.record();
    time.reStart();
    Tp(sim.getPos());
    // qprint("here");
    gui.scatter2D(point, dym::gi(6, 133, 135), 0, 0, n3);
    gui.scatter2D(point, dym::gi(237, 85, 59), 0, n3, 2 * n3);
    gui.scatter2D(point, dym::gi(255, 255, 255), 0, 2 * n3);  //
    // qprint("here");
    for (int i = 0; i < steps; ++i) sim.advance(dt);

    // qprint("here");
    // qprint(sim.x[100]);
  });
  return 0;
}

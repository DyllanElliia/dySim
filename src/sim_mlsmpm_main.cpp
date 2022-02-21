/*
 * @Author: DyllanElliia
 * @Date: 2022-02-16 15:30:48
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-17 17:45:35
 * @Description:
 */
#define DYM_DEFAULT_THREAD 1
#include <dySimulator.hpp>
#include <dyGraphic.hpp>
#include <random>
typedef dym::Vector<Real, 3> Vector3;

int main(int argc, char const *argv[]) {
  dym::init(dym::MidGrid);
  dym::setGlobalForce(Vector3({0.f, -9.8 * 2.f, 0.f}));
  std::default_random_engine re;
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  dym::Tensor<Vector3> asdf(0, dym::gi(5000));
  int particles_num = dym::addParticle(asdf.for_each_i([&](Vector3 &v) {
    v = Vector3({u(re), u(re), u(re)}) * 0.2f + Vector3({0.5, 0.3, 0.5});
  }),
                                       dym::addLiquidMateria(0, 0.f));
  dym::GUI gui("simMpm-test", dym::gi(17, 47, 65));
  gui.init(500, 500);
  const Real dt = 5e-5;
  dym::Tensor<dym::Vector<Real, 2>> point(0, dym::gi(particles_num));
  const int steps = 25, n3 = particles_num / 3;
  auto Tp = [&](dym::Tensor<Vector3> &x) {
    point.for_each_i([&](dym::Vector<Real, 2> &e, int i) {
      auto &pos = x[i];
      if (pos[0] <= 0 || pos[1] <= 0 || pos[2] <= 0 || pos[0] >= 1 ||
          pos[1] >= 1 || pos[2] >= 1)
        qprint(pos);
      float PI = 3.14159265f;
      float phi = (28.0 / 180) * PI, theta = (32.0 / 180) * PI;
      Vector3 a = pos / 1.5f - 0.35f;
      float c = std::cos(phi), s = std::sin(phi), C = std::cos(theta),
            S = std::sin(theta);
      float x = a.x() * c + a.z() * s, z = a.z() * c - a.x() * s;
      float u = x, v = a.y() * C + z * S;
      e[0] = 2 * u, e[1] = 2 * v;
    });
  };
  dym::TimeLog time;
  qprint(particles_num, n3);
  gui.update([&]() {
    // time.record();
    // time.reStart();
    Tp(dym::getPos());
    // qprint("here");
    gui.scatter2D(point, dym::gi(6, 133, 135), 0, 0, particles_num);
    // qprint("here");
    for (int i = 0; i < steps; ++i) dym::advance(dt);

    // qprint("here");
    // qprint(sim.x[100]);
  });
  return 0;
}

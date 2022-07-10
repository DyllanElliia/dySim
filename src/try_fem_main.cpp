/*
 * @Author: DyllanElliia
 * @Date: 2022-06-20 17:03:28
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-08 17:20:44
 * @Description:
 */
#include "render/BVH/bvhNode.hpp"
#include "render/object/triangle.hpp"
#include <dyGraphic.hpp>
#include <dyMath.hpp>
#include <dyRender.hpp>
#include <memory>

const int N = 2;
const Real dt = 5e-5;
const Real dx = 1.0 / N;
const Real rho = 4e1;
const int NF = 2 * dym::sqr(N);
const int NV = dym::pow(N + 1, 2);
const Real E = 4e4, nu = 0.2;
const Real mu = E / 2 / (1 + nu), lam = E * nu / (1 + nu) / (1 - 2 * nu);

dym::Vector2 ball_pos({0.5, 0.0});
const Real ball_radius = 0.31;
Real damping = 14.5;

dym::Tensor<dym::Vector2> pos(0, dym::gi(NV)), f(0, dym::gi(NV));
dym::Tensor<dym::Vector2> vel(0, dym::gi(NV));
dym::Tensor<dym::Vector3i> f2v(0, dym::gi(NF));
dym::Tensor<dym::Matrix<Real, 2, 2>> B(0, dym::gi(NF));
dym::Tensor<dym::Matrix<Real, 2, 2>> F(0, dym::gi(NF));
dym::Tensor<Real> V(0, dym::gi(NF)), phi(0, dym::gi(NF));
const auto I = dym::matrix::identity<Real, 2>(1.0);

dym::Vector2 gravity({0, -1}), attractor_pos({0, 0}),
    attractor_strength({0, 0});

bool showit = false;
_DYM_FORCE_INLINE_ void update_f() {
  f = 0.0;
  F.for_each_i([&](dym::Matrix<Real, 2, 2> &Fi, int i) {
    const auto &ia = f2v[i][0], &ib = f2v[i][1], &ic = f2v[i][2];
    auto &a = pos[ia], &b = pos[ib], &c = pos[ic];
    auto ac = a - c, bc = b - c;
    V[i] = dym::abs(((ac).cross(bc)).length());
    auto Aref = V[i] / 2.0;
    auto D_i = dym::Matrix<Real, 2, 2>({ac, bc}).transpose();
    Fi = D_i * B[i];
    auto G = (Fi.transpose() * Fi - I) / 2.0;
    auto S = 2 * mu * G + lam * G.trace() * I;
    auto f1 = -Aref * Fi * S * B[i][0], f2 = -Aref * Fi * S * B[i][1];
    auto f3 = -f1 - f2;
    f[ia] += f1, f[ib] += f2, f[ic] += f3;
  });
}

void advance(Real dt) {
  vel.for_each_i([&](dym::Vector2 &vi, int i) {
    auto acc = f[i];
    auto g = gravity * 30;
    vi += dt * (acc + g);
    vi *= dym::exp(-dt * damping);
  });
  pos.for_each_i([&](dym::Vector2 &posi, int i) {
    // Solve Hit Sphere
    auto disp = (posi - ball_pos);
    auto disp2 = disp.length_sqr();
    if (disp2 <= dym::sqr(ball_radius)) {
      auto NoV = vel[i].dot(disp);
      if (NoV < 0)
        vel[i] -= NoV * disp / disp2;
    }
    // Solve Rect boundary condition
    dym::Loop<int, 2>([&](auto j) {
      if (posi[j] < 0 && vel[i][j] < 0 || posi[j] > 1 && vel[i][j] > 0)
        vel[i][j] = 0;
    });
    posi += dt * vel[i];
  });
}

void init_pos() {
  for (int i = 0; i < N + 1; ++i)
    for (int j = 0; j < N + 1; ++j) {
      auto k = i * (N + 1) + j;
      pos[k] = dym::Vector2{(Real)i, (Real)j} / (Real)N * 0.25 + 0.45;
      vel[k] = 0;
      // qprint(pos[k]);
    }
  for (int i = 0; i < NF; ++i) {
    const auto &ia = f2v[i][0], &ib = f2v[i][1], &ic = f2v[i][2];
    auto &a = pos[ia], &b = pos[ib], &c = pos[ic];
    B[i] = dym::Matrix<Real, 2, 2>({a - c, b - c}).transpose().inverse();
    qprint(i, f2v[i]);
    qprint(a, b, c);
    qprint(a - c, b - c);
    qprint(B[i]);
  }
}

void init_mesh() {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      auto k = (i * N + j) * 2;
      auto a = i * (N + 1) + j;
      auto b = a + 1;
      auto c = a + N + 2;
      auto d = a + N + 1;
      f2v[k + 0] = {a, b, c};
      f2v[k + 1] = {c, d, a};
    }
}

auto genWorld() {
  dym::rt::HittableList world, rt;
  for (int i = 0; i < NF; ++i) {
    const auto &ia = f2v[i][0], &ib = f2v[i][1], &ic = f2v[i][2];
    const auto &a = pos[ia], &b = pos[ib], &c = pos[ic];
    auto log_J_i = dym::log(F[i].det());
    auto phi_i = mu / 2.0 * ((F[i].transpose() * F[i]).trace() - 2);
    phi_i -= mu * log_J_i;
    phi_i += lam / 2.0 * dym::sqr(log_J_i);
    auto k = phi_i * (10 / E);
    auto gb = (1 - k) * 0.5;
    dym::Vector3 tnormal({0, 0, 1});
    dym::rt::Vertex v0(dym::rt::Point3(a), tnormal, 0, 0),
        v1(dym::rt::Point3(b), tnormal, 0, 1),
        v2(dym::rt::Point3(c), tnormal, 1, 1);
    world.addObject<dym::rt::Triangle>(v0, v1, v2,
                                       std::make_shared<dym::rt::DiffuseLight>(
                                           dym::rt::ColorRGB{k + gb, gb, gb}));
  }
  rt.addObject<dym::rt::BvhNode>(world);
  return rt;
}

int main(int argc, char const *argv[]) {
  dym::GUI gui("fem", dym::gi(0, 0, 0));
  gui.init(800, 800);
  init_mesh();
  qprint("fin im");
  init_pos();
  qprint("fin ip");

  qprint(dym::Vector2{1, 0}.cross(dym::Vector2{0, 1}));

  dym::rt::Point3 lookfrom({0.5, 0.5, -1.35});
  dym::rt::Point3 lookat({0.5, 0.5, 0});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.0;

  dym::rt::RtRender render(500, 500);

  render.cam.setCamera(lookfrom, lookat, vup, 40, 1.f, aperture, dist_to_focus);

  gui.update([&]() {
    dym::TimeLog t;
    // getchar();
    for (int i = 0; i < 10; ++i) {
      update_f();
      advance(dt);
    }

    // qprint(pos[0], V[0], f[0], F[0], vel[0]);
    // qprint("1");
    gui.scatter2D(pos, dym::gi(255, 170, 51));
    // render.worlds = genWorld();
    // render.render(1, 2);
    // auto &image = render.getFrame();
    // gui.imshow(image);
  });

  return 0;
}

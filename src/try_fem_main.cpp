/*
 * @Author: DyllanElliia
 * @Date: 2022-06-20 17:03:28
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-13 17:25:38
 * @Description:
 */
#include <dyGraphic.hpp>
#include <dyMath.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>

const int N = 3;
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
const auto I = dym::matrix::identity<Real, 2>(1.0);

dym::Vector2 gravity({0, -1});

dym::Dual<Real> StVK(dym::Matrix<dym::Dual<Real>, 2, 2> G) {
  auto &euu = G[0][0], euv = G[0][1] + G[1][0], &evv = G[1][1];
  euv.A /= 2.0;
  return lam / 2.0 * dym::sqr(euu + evv) +
         mu * (dym::sqr(euu) + dym::sqr(euv) + dym::sqr(evv));
}

_DYM_FORCE_INLINE_ void update_f() {
  f = 0.0;
  F.for_each_i([&](dym::Matrix<Real, 2, 2> &Fi, int i) {
    const auto &ia = f2v[i][0], &ib = f2v[i][1], &ic = f2v[i][2];
    auto &a = pos[ia], &b = pos[ib], &c = pos[ic];
    auto ac = a - c, bc = b - c;
    auto Aref = dym::abs(((ac).cross(bc)).length()) / 2.0;
    auto D_i = dym::Matrix<Real, 2, 2>({ac, bc}).transpose();
    Fi = D_i * B[i];
    auto G = (Fi.transpose() * Fi - I) / 2.0;
    dym::Matrix<dym::Dual<Real>, 2, 2> dG(
        [&](auto &dGi, int i, int j) { dGi = G[i][j]; });
    auto S = dym::AD::dx(StVK, dG, dym::AD::all(dG));
    // auto S = 2 * mu * G + lam * G.trace() * I;
    auto f1 = -Aref * Fi * S * B[i][0], f2 = -Aref * Fi * S * B[i][1];
    auto f3 = -f1 - f2;
    f[ia] += f1, f[ib] += f2, f[ic] += f3;
  });
}

void advance(Real dt) {
  vel.for_each_i([&](dym::Vector2 &vi, int i) {
    vi += dt * (f[i] + gravity * 30);
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
    }
  for (int i = 0; i < NF; ++i) {
    const auto &ia = f2v[i][0], &ib = f2v[i][1], &ic = f2v[i][2];
    auto &a = pos[ia], &b = pos[ib], &c = pos[ic];
    B[i] = dym::Matrix<Real, 2, 2>({a - c, b - c}).transpose().inverse();
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

std::vector<dym::rt::Vertex> vpool(NV);
auto genWorld() {
  dym::rt::HittableList world, rt;
  pos.for_each_i([&](dym::Vector2 &p, int i) {
    vpool[i].point = p, vpool[i].normal = {0, 0, -1};
  });
  for (int i = 0; i < NF; ++i) {
    const auto &ia = f2v[i][0], &ib = f2v[i][1], &ic = f2v[i][2];
    auto log_J_i = dym::log(F[i].det());
    auto phi_i = mu / 2.0 * ((F[i].transpose() * F[i]).trace() - 2);
    phi_i += phi_i += lam / 2.0 * dym::sqr(log_J_i) - mu * log_J_i;
    auto k = phi_i * (10 / E);
    auto gb = (1 - k) * 0.5;
    world.add(std::make_shared<dym::rt::Triangle>(
        vpool[ia], vpool[ib], vpool[ic],
        std::make_shared<dym::rt::DiffuseLight>(
            dym::min(dym::rt::ColorRGB{k + gb, gb, gb}, {1, 1, 1}))));
  }
  auto yellowPoint =
      std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB{1, 0.6, 0.2});
  for (int i = 0; i < NV; ++i)
    world.addObject<dym::rt::Sphere>(dym::Vector3(pos[i]), dx / 50.0,
                                     yellowPoint);
  world.addObject<dym::rt::Sphere>(
      ball_pos, ball_radius,
      std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB(0.5)));
  rt.addObject<dym::rt::BvhNode>(world);
  return rt;
}

int main(int argc, char const *argv[]) {
  dym::GUI gui("fem", dym::gi(0, 0, 0));
  gui.init(800, 800);
  init_mesh();
  init_pos();
  dym::rt::Point3 lookfrom({0.5, 0.5, -1.35});
  dym::rt::Point3 lookat({0.5, 0.5, -0.1});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.0;
  dym::rt::RtRender render(500, 500);
  render.cam.setCamera(lookfrom, lookat, vup, 40, 1.f, aperture, dist_to_focus);

  int ccc = 1;
  gui.update([&]() {
    dym::TimeLog t;
    for (int i = 0; i < 50; ++i) {
      update_f();
      advance(dt);
    }
    render.worlds = genWorld();
    render.render(1, 2, [&](const dym::rt::Ray &r) {
      return dym::Vector3{0.1, 0.15, 0.1};
    });
    auto &image = render.getFrame();
    // dym::imwrite(image, "./fem_out/sample/1/frame_" +
    //                         std::to_string(ccc++ - 1) + ".jpg");
    gui.imshow(image);
  });

  return 0;
}

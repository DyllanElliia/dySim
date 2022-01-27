/*
 * @Author: DyllanElliia
 * @Date: 2022-01-27 15:57:53
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-01-27 18:36:05
 * @Description:
 */
/*
 * @Author: DyllanElliia
 * @Date: 2021-11-18 16:16:19
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-12-08 15:25:14
 */
//  88-Line 2D Moving Least Squares Material Point Method (MLS-MPM)
// [Explained Version by David Medina]

// Uncomment this line for image exporting functionality
#define TC_IMAGE_IO

// Note: You DO NOT have to install taichi or taichi_mpm.
// You only need [taichi.h] - see below for instructions.
#include "tiSvd.hpp"

#include <cmath>

using namespace taichi;
using Vec = Vector3;
using Mat = Matrix3;

// Window
const int window_size = 800;

// Grid resolution (cells)
const int n = 32;

const int dim = 3;

const real dt = 1e-4_f;
const real frame_dt = 10e-4_f;
const real dx = 1.0_f / n;
const real inv_dx = real(n);
const real gravity = 9.8 * 2;

const real max_v = 100;
const real max_x = 0;
const int bound = 3;

// Snow material properties
const real p_rho = 1.0_f;
const auto p_vol = (dx * dx * 0.25_f);  // Particle Volume
const auto p_mass = p_vol * p_rho;
#define hardening 10.0_f  // Snow hardening factor
const auto E = 0.1e4_f;   // Young's Modulus
const auto nu = 0.2_f;    // Poisson ratio

// Initial Lamé parameters
const real mu_0 = E / (2 * (1 + nu));
const real lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

struct Particle {
  // Position and velocity
  Vec x, v;
  // Deformation gradient
  Mat F;
  // Affine momentum from APIC
  Mat C;
  // Determinant of the deformation gradient (i.e. volume)
  real Jp;
  // Color
  int c;
  int material_i;

  Particle(Vec x, int c, int material_i_, Vec v = Vec(0))
      : x(x), v(v), F(1), C(0), Jp(1), c(c), material_i(material_i_) {}
};

std::vector<Particle> particles;

Vector3 grid_v[n + 1][n + 1][n + 1];
real grid_m[n + 1][n + 1][n + 1];

#define m_LIQUID int(0)
#define m_JELLY int(1)
#define m_PLASTIC int(2)
#define m_SAND int(3)
#define m_SOIL int(4)
std::array<std::function<void(real &, real &, Particle &p)>, 5> c_mu_la = {
    // Liquid
    [](real &mu, real &lambda, Particle &p) {
      auto h = exp(hardening * (1.0f - p.Jp));
      mu = 0.f, lambda = lambda_0 * h;
    },
    // Jelly
    [](real &mu, real &lambda, Particle &p) {
      auto h = 0.3f;
      mu = mu_0 * h, lambda = lambda_0 * h;
    },
    // Plastic
    [](real &mu, real &lambda, Particle &p) {
      auto h = exp(hardening * (1.0f - p.Jp));
      mu = mu_0 * h, lambda = lambda_0 * h;
    },
    // Sand
    [](real &mu, real &lambda, Particle &p) {
      auto h = exp(hardening * (1.0f - p.Jp));
      mu = mu_0 * h, lambda = lambda_0 * h;
    },
    // Soil
    [](real &mu, real &lambda, Particle &p) {
      auto h = exp(hardening * (1.0f - p.Jp));
      mu = mu_0 * h, lambda = lambda_0 * h;
    }};

std::array<std::function<void(real &)>, 5> yield_criteria = {
    // Liquid
    [](real &new_Sig) {},
    // Jelly
    [](real &new_Sig) {},
    // Plastic
    [](real &new_Sig) {
      new_Sig = taichi::clamp(new_Sig, 1.f - 2.5e-2f, 1.f + 8.5e-3f);
    },
    // Sand
    [](real &new_Sig) {
      new_Sig = taichi::clamp(new_Sig, 1.f - 2.5e-2f, 1.f + 8.5e-3f);
    },
    // Soil
    [](real &new_Sig) {
      new_Sig = taichi::clamp(new_Sig, 1.f - 2.5e-2f, 1.f + 8.5e-3f);
    }};
int cnt = 0;
std::array<std::function<void(Mat &, Mat &, Mat &, Mat &)>, 5> c_F = {
    // Liquid
    [](Mat &F, Mat &U, Mat &Sig, Mat &V) {},
    // Jelly
    [](Mat &F, Mat &U, Mat &Sig, Mat &V) {},
    // Plastic
    [](Mat &F, Mat &U, Mat &Sig, Mat &V) {
      F = U * Sig * transposed(V);
      // if ((cnt++) == 1000) {
      //   cout << U << Sig << transposed(V) << F << endl, cnt = 0;
      //   getchar();
      // }
    },
    // Sand
    [](Mat &F, Mat &U, Mat &Sig, Mat &V) {
      F = U * Sig * transposed(V);
      // if ((cnt++) == 1000) {
      //   cout << U << Sig << transposed(V) << F << endl, cnt = 0;
      //   getchar();
      // }
    },
    // Soil
    [](Mat &F, Mat &U, Mat &Sig, Mat &V) {
      F = U * Sig * transposed(V);
      // if ((cnt++) == 1000) {
      //   cout << U << Sig << transposed(V) << F << endl, cnt = 0;
      //   getchar();
      // }
    }};

inline Vec RotateField(const Vec &center, const Vec &loc, const real &w) {
  return Vec(0);
}

inline Vec solveMovingBound(const Vec &gV, const Vec &bN, const Vec &bV) {
  return gV - bN * std::min(dot(bN, gV - bV), 1e-4f);
}

inline bool is_inShpere(const Vec &center, const real &r, const Vec &loc) {
  return length(loc - center) < r;
}

inline Vec getSphereN(const Vec &center, const Vec &loc) {
  return normalize(loc - center);
}

void advance(real dt) {
  // Reset grid
  std::memset(grid_v, 0, sizeof(grid_v));
  std::memset(grid_m, 0, sizeof(grid_m));
  int hn = n >> 1;
  int cnt = 0;

  // P2G
  for (auto &p : particles) {
    auto Xp = p.x / dx;
    auto base = (Xp - Vec(0.5f)).cast<int>();
    Vec fx = Xp - base.cast<real>();
    std::array<Vec, 3> w = {Vec(0.50f) * sqr(Vec(1.5f) - fx),
                            Vec(0.75f) - sqr(fx - Vec(1.0f)),
                            Vec(0.50f) * sqr(fx - Vec(0.5f))};
    p.F = (Mat(1) + dt * p.C) * p.F;
    Mat U, Sig, V;
    svd(p.F, U, Sig, V);
    real mu, lambda;
    c_mu_la[p.material_i](mu, lambda, p);

    real J = 1.0f;
    // std::cout << "cSig" << endl;
    auto &y_c_f = yield_criteria[p.material_i];
    for (int d = 0; d < 3; ++d) {
      auto new_Sig = Sig[d][d];
      y_c_f(new_Sig);
      p.Jp *= Sig[d][d] / new_Sig;
      Sig[d][d] = new_Sig;
      J *= new_Sig;
    }
    c_F[p.material_i](p.F, U, Sig, V);

    Mat stress = 2 * mu * ((p.F - U * transposed(V)) * transposed(p.F)) +
                 Mat(1) * lambda * J * (J - 1);
    stress = (-dt * p_vol * 4) * stress * sqr(inv_dx);
    Mat affine = stress + p_mass * p.C;
    // std::cout << "P2G" + std::to_string(++cnt) << endl;

    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
          Vec dpos = (Vec(i, j, k) - fx) * dx;
          real weight = w[i].x * w[j].y * w[k].z;
          grid_v[base.x + i][base.y + j][base.z + k] +=
              weight * (p_mass * p.v + affine * dpos);
          grid_m[base.x + i][base.y + j][base.z + k] += weight * p_mass;
        }
  }

  for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j)
      for (int k = 0; k <= n; ++k) {
        auto &g_v = grid_v[i][j][k];
        auto &g_m = grid_m[i][j][k];
        if (g_m > 0) g_v /= g_m;
        g_v.y -= dt * gravity;
        // Vec loc(i, j, k);
        // loc /= real(n);
        // Vec center(0.5, 0.1, 0.5);
        // real R = 0.15;
        // if (is_inShpere(center, R, loc))
        //   g_v = solveMovingBound(g_v, getSphereN(center, loc), Vec(0));
        if ((i < bound || k < bound) ||
            (i > n - bound || j > n - bound || k > n - bound)) {
          g_v = Vec(0);
          // cout << "run" << endl;
        }
        if (j < bound) g_v[1] = std::max(0.0f, g_v[1]);
      }
  int iii = 0;
  for (auto &p : particles) {
    // cnt++;
    auto Xp = p.x / dx;
    auto base = (Xp - Vec(0.5f)).cast<int>();
    Vec fx = Xp - base.cast<real>();
    std::array<Vec, 3> w = {Vec(0.50f) * sqr(Vec(1.5f) - fx),
                            Vec(0.75f) - sqr(fx - Vec(1.0f)),
                            Vec(0.50f) * sqr(fx - Vec(0.5f))};
    Vec new_v(0);
    Mat new_C(0);
    if (iii == 100) std::cout << "begin " << new_C;
    auto nc_d = 4 * sqr(inv_dx);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
          Vec dpos = (Vec(i, j, k) - fx) * dx;
          if (iii == 100) {
            std::cout << dpos << base << fx;
          }
          real weight = w[i].x * w[j].y * w[k].z;
          auto &g_v = grid_v[base.x + i][base.y + j][base.z + k];
          new_v += weight * g_v;
          new_C += nc_d * weight * Mat::outer_product(g_v, dpos);
          if (iii == 100 && i == 1 && j == 1 && k == 1) {
            std::cout << i << " " << j << " " << k << std::endl;
            std::cout << g_v << dpos << Mat::outer_product(g_v, dpos)
                      << nc_d * weight * Mat::outer_product(g_v, dpos);
            ;
          }
        }
    iii++;
    // if (cnt == 32) std::cout << p.x << p.v << new_v << new_C << endl;
    p.v = new_v, p.C = new_C;
    // p.v.x = taichi::clamp(p.v.x, -max_v, max_v);
    // p.v.y = taichi::clamp(p.v.y, -max_v, max_v);
    // p.v.z = taichi::clamp(p.v.z, -max_v, max_v);
    p.x += dt * p.v;

    // p.x.x = taichi::clamp(p.x.x, max_x, 1 - max_x);
    // p.x.y = taichi::clamp(p.x.y, max_x, 1 - max_x);
    // p.x.z = taichi::clamp(p.x.z, max_x, 1 - max_x);
    // if (cnt == 32) cout << p.x << "end" << endl;
  }
  // cnt = 0;
  // cout << particles[100].x << ":" << particles[100].v << endl;
  // cout << grid_v[hn][hn][hn] << endl;
  // getchar();
}

// Seed particles with position and color
void add_object(Vec center, int c, int material_i) {
  int p_n = std::pow(n, dim) / std::pow(2, (dim - 1));
  // Randomly sample 1000 particles in the square
  for (int i = 0; i < 1000; i++) {
    particles.push_back(
        Particle((Vec::rand() * 2.0f - Vec(1)) * 0.1f + center, c, material_i));
  }
}

Vector2 T(Vec a) {
  float PI = 3.14159265f;
  float phi = (28.0 / 180) * PI, theta = (32.0 / 180) * PI;
  a = Vec(a.x / 1.5 - 0.35, a.y / 1.5 - 0.35, a.z / 1.5 - 0.35);
  float c = cos(phi), s = sin(phi), C = cos(theta), S = sin(theta);
  float x = a.x * c + a.z * s, z = a.z * c - a.x * s;
  float u = x + 0.5, v = a.y * C + z * S + 0.5;
  return Vector2(u, v);
}

int main() {
  GUI gui("Real-time 3D MLS-MPM", window_size, window_size);
  auto &canvas = gui.get_canvas();

  // add_object(Vec(0.5, 0.5, 0.5), 0xED553B);
  // add_object(Vec(0.25, 0.25, 0.5), 0x068587, m_LIQUID);
  // add_object(Vec(0.25, 0.45, 0.5), 0x068587, m_LIQUID);
  // add_object(Vec(0.25, 0.65, 0.5), 0x068587, m_LIQUID);
  // add_object(Vec(0.25, 0.85, 0.5), 0x068587, m_LIQUID);
  // add_object(Vec(0.75, 0.25, 0.5), 0x068587, m_LIQUID);
  // add_object(Vec(0.75, 0.45, 0.5), 0x068587, m_LIQUID);
  // add_object(Vec(0.75, 0.65, 0.5), 0x068587, m_LIQUID);
  // add_object(Vec(0.75, 0.85, 0.5), 0x068587, m_LIQUID);

  add_object(Vec(0.25, 0.25, 0.5), 0x068587, m_LIQUID);
  add_object(Vec(0.45, 0.45, 0.5), 0xED553B, m_JELLY);
  add_object(Vec(0.65, 0.65, 0.5), 0xFFFFFF, m_PLASTIC);

  int frame = 0;

  std::cout << Mat::outer_product(Vec(1, 2, 3), Vec(4, 5, 6));
  std::cout << (Mat(2) + Mat(3).transposed()) * Vec({1, 2, 3});

  auto Xp = Vec(0.1, 0.2, 0.3) / dx;
  auto base = (Xp - Vec(0.5f)).cast<int>();
  Vec fx = Xp - base.cast<real>();
  Vec dpos = (Vec(0, 1, 2) - fx) * dx;
  std::array<Vec, 3> w = {Vec(0.50f) * sqr(Vec(1.5f) - fx),
                          Vec(0.75f) - sqr(fx - Vec(1.0f)),
                          Vec(0.50f) * sqr(fx - Vec(0.5f))};
  std::cout << w[0] << w[1] << w[2] << std::endl
            << "dpos:" << Xp << base << fx << dpos << std::endl;

  std::cout << particles[0].v << particles[0].C << particles[0].F
            << particles[0].Jp << std::endl;
  particles[0].x = Vec(0.25, 0.25, 0.5);

  Mat pF(2.0f), pC(0);
  pC[0][0] = 1;
  Mat U, Sig, V;
  svd(pF, U, Sig, V);
  real mu, lambda;
  c_mu_la[0](mu, lambda, particles[0]);

  real J = 1.0f;
  // std::cout << "cSig" << endl;
  auto &y_c_f = yield_criteria[0];
  for (int d = 0; d < 3; ++d) {
    auto new_Sig = Sig[d][d];
    y_c_f(new_Sig);
    // p.Jp *= Sig[d][d] / new_Sig;
    Sig[d][d] = new_Sig;
    J *= new_Sig;
  }
  c_F[0](pF, U, Sig, V);

  Mat stress = 2 * mu * ((pF - U * transposed(V)) * transposed(pF)) +
               Mat(1) * lambda * J * (J - 1);
  std::cout << Mat(1) * lambda * J * (J - 1) << E << lambda << J << stress
            << std::endl;
  stress = (-dt * p_vol * 4) * stress * sqr(inv_dx);
  Mat affine = stress + p_mass * pC;
  std::cout << "asdf" << pF << pC << stress << affine << std::endl;

  getchar();
  // Main Loop
  for (int step = 0;; step++) {
    // Advance simulation
    advance(dt);
    // getchar();
    // Visualize frame
    if (step % int(frame_dt / dt) == 0) {
      // Clear background
      canvas.clear(0x112F41);
      // Box
      canvas.rect(Vector2(0.04), Vector2(0.96))
          .radius(2)
          .color(0x4FB99F)
          .close();
      // Particles
      for (auto p : particles) {
        canvas.circle(T(p.x)).radius(2).color(p.c);
      }
      // Update image
      gui.update();

      std::cout << particles[100].x << particles[100].F << particles[100].C;
      getchar();

      // Write to disk (optional)
      // canvas.img.write_as_image(fmt::format("tmp/{:05d}.png", frame++));
    }
  }
}

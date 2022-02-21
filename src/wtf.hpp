/*
 * @Author: DyllanElliia
 * @Date: 2022-02-17 15:45:37
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-17 15:45:38
 * @Description:
 */
#define DYM_DEFAULT_THREAD 1
#include <dyGraphic.hpp>
#include <dyMath.hpp>
#include <dyPicture.hpp>
#include <random>

typedef unsigned int u_int;
#define cconst constexpr const

const u_int dim = 3, n_grid = 48, steps = 25;
const Real dt = 1e-4f;

const u_int n_particles = std::pow(n_grid, dim) / (std::pow(2, dim - 1));
const Real dx = 1 / Real(n_grid), inv_dx = Real(n_grid);

const Real p_rho = 1, p_vol = std::pow(dx * 0.5, 2), p_mass = p_vol * p_rho,
           gravity = 9.8 * 2, hardening = 10.f;
const int bound = 3;
const Real E = 0.1e4f, nu = 0.2f;
const Real mu_0 = E / (2 * (1 + nu)),
           lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

typedef dym::Matrix<Real, 3, 3> Matrix3;
typedef dym::Vector<Real, 3> Vector3;
typedef dym::Vector<Real, 4> Vector4;
typedef dym::Vector<int, 3> Vector3i;

#define identity3 dym::matrix::identity<Real, dim>()

struct Particle_o {
  Vector3 v;
  Matrix3 C, F;
  Real Jp;
  unsigned short material;
  Particle_o() : v(0), C(0), F(identity3), Jp(1.f), material(0) {}
};
Particle_o pdefault;

dym::Tensor<Vector3> x(0, dym::gi(n_particles));
auto particles = new Particle_o[n_particles];

bool dynamic_create = true;
auto pExists = new bool[n_particles];
dym::Tensor<Vector4> grid_vm(0, dym::gi(n_grid + 1, n_grid + 1, n_grid + 1));

#define m_LIQUID int(0)
#define m_JELLY int(1)
#define m_PLASTIC int(2)
#define m_SAND int(3)
#define m_SOIL int(4)

std::array<std::function<void(Real &, Real &, Particle_o &p)>, 5> c_mu_la = {
    // Liquid
    [](Real &mu, Real &lambda, Particle_o &p) {
      auto h = exp(hardening * (1.0f - p.Jp));
      mu = 0.f, lambda = lambda_0 * h;
    },
    // Jelly
    [](Real &mu, Real &lambda, Particle_o &p) {
      auto h = 0.3f;
      mu = mu_0 * h, lambda = lambda_0 * h;
    },
    // Plastic
    [](Real &mu, Real &lambda, Particle_o &p) {
      auto h = exp(hardening * (1.0f - p.Jp));
      mu = mu_0 * h, lambda = lambda_0 * h;
    },
    // Sand
    [](Real &mu, Real &lambda, Particle_o &p) {
      auto h = exp(hardening * (1.0f - p.Jp));
      mu = mu_0 * h, lambda = lambda_0 * h;
    },
    // Soil
    [](Real &mu, Real &lambda, Particle_o &p) {
      auto h = exp(hardening * (1.0f - p.Jp));
      mu = mu_0 * h, lambda = lambda_0 * h;
    }};

std::array<std::function<void(Real &)>, 5> yield_criteria = {
    // Liquid
    [](Real &new_Sig) {},
    // Jelly
    [](Real &new_Sig) {},
    // Plastic
    [](Real &new_Sig) {
      new_Sig = dym::clamp(new_Sig, 1.f - 2.5e-2f, 1.f + 8.5e-3f);
    },
    // Sand
    [](Real &new_Sig) {
      new_Sig = dym::clamp(new_Sig, 1.f - 2.5e-2f, 1.f + 8.5e-3f);
    },
    // Soil
    [](Real &new_Sig) {
      new_Sig = dym::clamp(new_Sig, 1.f - 2.5e-2f, 1.f + 8.5e-3f);
    }};

std::array<std::function<void(Matrix3 &, Matrix3 &, Matrix3 &, Matrix3 &)>, 5>
    c_F = {
        // Liquid
        [](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {},
        // Jelly
        [](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {},
        // Plastic
        [](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {
          F = U * Sig * V.transpose();
        },
        // Sand
        [](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {
          F = U * Sig * V.transpose();
        },
        // Soil
        [](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {
          F = U * Sig * V.transpose();
        }};

inline bool is_beCreated(const u_int &i) { return true; }

void advance(const Real &dt) {
  grid_vm = Vector4(0.f);
  x.for_each_i([&](Vector3 &px, int pi) {
    auto Xp = px / dx;
    auto base = (Xp - Vector3(0.5f)).cast<int>();
    Vector3 fx = Xp - base.cast<Real>();
    std::array<Vector3, 3> w = {Vector3(0.50f) * dym::sqr(Vector3(1.5f) - fx),
                                Vector3(0.75f) - dym::sqr(fx - Vector3(1.0f)),
                                Vector3(0.50f) * dym::sqr(fx - Vector3(0.5f))};
    auto &p = particles[pi];
    p.F = (identity3 + dt * p.C) * p.F;
    Matrix3 U, Sig, V;
    dym::matrix::svd(p.F, U, Sig, V);
    Real mu, lambda;
    c_mu_la[p.material](mu, lambda, p);
    Real J = 1.f;
    auto &y_c_f = yield_criteria[p.material];
    for (int d = 0; d < dim; ++d) {
      auto new_Sig = Sig[d][d];
      y_c_f(new_Sig);
      p.Jp *= Sig[d][d] / new_Sig;
      Sig[d][d] = new_Sig, J *= new_Sig;
    }
    c_F[p.material](p.F, U, Sig, V);
    Matrix3 stress = 2.f * mu * ((p.F - U * V.transpose()) * p.F.transpose()) +
                     identity3 * lambda * J * (J - 1.f);
    stress = (-dt * p_vol * 4) * stress * dym::sqr(inv_dx);
    Matrix3 affine = stress + p_mass * p.C;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        for (int k = 0; k < dim; ++k) {
          Vector3 dpos = (Vector3({Real(i), Real(j), Real(k)}) - fx) * dx;
          Real weight = w[i].x() * w[j].y() * w[k].z();
          grid_vm[dym::gi(base.x() + i, base.y() + j, base.z() + k)] +=
              Vector4(weight * (p_mass * p.v + affine * dpos), weight * p_mass);
        }
  });
  grid_vm.for_each_i([&](Vector4 &gvm, int i, int j, int k) {
    auto &g_m = gvm[dim];
    if (g_m > 0.f) {
      gvm /= g_m;
      gvm[1] -= dt * gravity;
      if ((i < bound || k < bound) ||
          (i > n_grid - bound || j > n_grid - bound || k > n_grid - bound))
        gvm = 0;
      if (j < bound) gvm[1] = std::max(gvm[1], 0.f);
    }
  });
  x.for_each_i([&](Vector3 &px, int pi) {
    auto Xp = px / dx;
    auto base = (Xp - Vector3(0.5f)).cast<int>();
    Vector3 fx = Xp - base.cast<Real>();
    std::array<Vector3, 3> w = {Vector3(0.50f) * dym::sqr(Vector3(1.5f) - fx),
                                Vector3(0.75f) - dym::sqr(fx - Vector3(1.0f)),
                                Vector3(0.50f) * dym::sqr(fx - Vector3(0.5f))};
    auto &p = particles[pi];
    Vector3 new_v(0.f);
    Matrix3 new_C(0.f);
    auto nc_d = 4 * dym::sqr(inv_dx);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        for (int k = 0; k < dim; ++k) {
          Vector3 dpos = (Vector3({Real(i), Real(j), Real(k)}) - fx) * dx;
          Real weight = w[i].x() * w[j].y() * w[k].z();
          auto g_v = Vector3(
              grid_vm[dym::gi(base.x() + i, base.y() + j, base.z() + k)]);
          new_v += weight * g_v;
          new_C += (nc_d * weight) * dym::matrix::outer_product(g_v, dpos);
        }
    p.v = new_v, p.C = new_C;
    px += dt * p.v;
  });
}
std::default_random_engine re;
std::uniform_real_distribution<float> u(-1.f, 1.f);
void add_object(Vector3 center, int begin, int end, int material_index) {
  for (int i = begin; i < end; ++i) {
    x[i] = Vector3({u(re), u(re), u(re)}) * 0.1f + center;
    particles[i].material = material_index;
  }
}
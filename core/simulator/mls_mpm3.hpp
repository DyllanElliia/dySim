/*
 * @Author: DyllanElliia
 * @Date: 2022-02-17 16:02:02
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-21 16:56:52
 * @Description:
 */
#pragma once
#include "../dyMath.hpp"

#include <random>

namespace dym {
namespace mpmt {

typedef unsigned int u_int;
#define cconst constexpr const
#define dim 3

u_int n_grid = 0, steps = 25;

u_int n_particles = 0;
Real dx = 0, inv_dx = 0;

int bound = n_grid / 16;

// const Real p_rho = 1, p_vol = std::pow(dx * 0.5, 2), p_mass = p_vol * p_rho,
//            gravity = 9.8 * 2, hardening = 10.f;

// const Real E = 0.1e4f, nu = 0.2f;
// const Real mu_0 = E / (2 * (1 + nu)),
//            lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

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

dym::Tensor<Vector3> x;
std::vector<Particle_o> particles;

dym::Tensor<Vector4> grid_vm;

#define m_LIQUID int(0)
#define m_JELLY int(1)
#define m_PLASTIC int(2)

std::array<std::function<void(Real &, Real &, const Particle_o &, const Real &,
                              const Real &, const Real &, const Real &)>,
           3>
    c_mu_la = {
        // Liquid
        [](Real &mu, Real &lambda, const Particle_o &p, const Real &hardening,
           const Real &mu_0, const Real &lambda_0, const Real &opt) {
          auto h = exp(hardening * (1.0f - p.Jp));
          mu = opt, lambda = lambda_0 * h;
        },
        // Jelly
        [](Real &mu, Real &lambda, const Particle_o &p, const Real &hardening,
           const Real &mu_0, const Real &lambda_0, const Real &opt) {
          auto h = opt;
          mu = mu_0 * h, lambda = lambda_0 * h;
        },
        // Plastic
        [](Real &mu, Real &lambda, const Particle_o &p, const Real &hardening,
           const Real &mu_0, const Real &lambda_0, const Real &opt) {
          auto h = exp(hardening * (1.0f - p.Jp));
          mu = mu_0 * h, lambda = lambda_0 * h;
        }};

std::array<std::function<void(Real &, const Real &, const Real &)>, 3>
    yield_criteria = {
        // Liquid
        [](Real &new_Sig, const Real &y_min, const Real &y_max) {},
        // Jelly
        [](Real &new_Sig, const Real &y_min, const Real &y_max) {},
        // Plastic
        [](Real &new_Sig, const Real &y_min, const Real &y_max) {
          new_Sig = dym::clamp(new_Sig, y_min, y_max);
        }};

std::array<std::function<void(Matrix3 &, Matrix3 &, Matrix3 &, Matrix3 &)>, 3>
    c_F = {
        // Liquid
        [](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {},
        // Jelly
        [](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {},
        // Plastic
        [](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {
          F = U * Sig * V.transpose();
        }};

enum SimGridAcc {
  LowGrid = (u_short)32,
  MidGrid = (u_short)48,
  HighGrid = (u_short)64
};

_DYM_FORCE_INLINE_ bool init(const SimGridAcc gridNum = SimGridAcc::MidGrid) {
  n_particles = 0, n_grid = gridNum;
  dx = 1 / Real(n_grid), inv_dx = Real(n_grid), bound = n_grid / 16;
  grid_vm.reShape(dym::gi(n_grid + 1, n_grid + 1, n_grid + 1));
  x.reShape(dym::gi(0));
  return true;
}

struct Material {
  Real p_rho, p_vol, p_mass, hardening, mu_0, lambda_0, y_min, y_max, opt;
  u_short mat_fun_i;
  Material(const Real &E, const Real &nu, const Real &hardening,
           const Real &p_rho, const Real &p_vol, const Real &y_min,
           const Real &y_max, const Real &opt, const u_short &mat_fun_i)
      : mu_0(E / (2 * (1 + nu))),
        lambda_0(E * nu / ((1 + nu) * (1 - 2 * nu))),
        hardening(hardening),
        p_rho(p_rho),
        p_vol(p_vol),
        p_mass(p_vol * p_rho),
        y_min(y_min),
        y_max(y_max),
        opt(opt),
        mat_fun_i(mat_fun_i) {
    qprint(this->p_rho, this->p_vol, this->p_mass, this->hardening, this->mu_0,
           this->lambda_0);
  }
};

std::vector<Material> material_l;
Vector3 globalForce;

_DYM_FORCE_INLINE_ u_int addMaterial(
    const u_short &material_model, const Real &E = 0.1e4f,
    const Real &nu = 0.2f, const Real &hardening = 10.f,
    const Real &y_min = 1.f - 2.5e-2f, const Real &y_max = 1.f + 8.5e-3f,
    const Real &opt = 0.f, const Real &p_rho = 1.f, Real p_vol = -1) {
  if (p_vol < 0) p_vol = dym::pow(dx * 0.5, 2);
  material_l.push_back(Material(E, nu, hardening, p_rho, p_vol, y_min, y_max,
                                opt, material_model));
  return material_l.size() - 1;
}

_DYM_FORCE_INLINE_ u_int addLiquidMaterial(
    const Real &mu = 0.f, const Real &E = 0.1e4f, const Real &nu = 0.2f,
    const Real &hardening = 10.f, const Real &p_rho = 1.f, Real p_vol = -1) {
  return addMaterial(0, E, nu, hardening, 1.f, 1.f, mu, p_rho, p_vol);
}

_DYM_FORCE_INLINE_ u_int addJellyMaterial(
    const Real &h = 0.3f, const Real &E = 0.1e4f, const Real &nu = 0.2f,
    const Real &hardening = 10.f, const Real &p_rho = 1.f, Real p_vol = -1) {
  return addMaterial(1, E, nu, hardening, 1.f, 1.f, h, p_rho, p_vol);
}

_DYM_FORCE_INLINE_ u_int addPlasticMaterial(
    const Real &y_min = 1.f - 2.5e-2f, const Real &y_max = 1.f + 8.5e-3f,
    const Real &E = 0.1e4f, const Real &nu = 0.2f, const Real &hardening = 10.f,
    const Real &p_rho = 1.f, Real p_vol = -1) {
  return addMaterial(2, E, nu, hardening, y_min, y_max, 0.f, p_rho, p_vol);
}

_DYM_FORCE_INLINE_ Vector3 addGlobalForce(const Vector3 &f) {
  globalForce += f;
  return globalForce;
}
_DYM_FORCE_INLINE_ Vector3 setGlobalForce(const Vector3 &f) {
  globalForce = f;
  return globalForce;
}
_DYM_FORCE_INLINE_ auto &getPos() { return x; }
_DYM_FORCE_INLINE_ auto getParticlesNum() { return n_particles; }

u_int addParticle(Tensor<Vector3> newX, const u_int &material_index) {
  qprint(material_index);
  int par_num = newX.shape()[0];
  Particle_o par;
  par.material = material_index;
  particles.insert(particles.end(), par_num, par);
  x.reShape(dym::gi(n_particles + par_num));
  newX.for_each_i([&](Vector3 &pos, int i) {
    if (pos > 1.f) qprint(i, pos);
    x[n_particles + i] = pos;
  });
  n_particles += par_num;
  return getParticlesNum();
}

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
    const auto &mat_data = material_l[p.material];
    const auto &p_vol = mat_data.p_vol, &p_mass = mat_data.p_mass;
    const auto &mat_fun_i = mat_data.mat_fun_i;
    p.F = (identity3 + dt * p.C) * p.F;
    Matrix3 U, Sig, V;
    dym::matrix::svd(p.F, U, Sig, V);
    Real mu, lambda;
    c_mu_la[mat_fun_i](mu, lambda, p, mat_data.hardening, mat_data.mu_0,
                       mat_data.lambda_0, mat_data.opt);
    Real J = 1.f;
    auto &y_c_f = yield_criteria[mat_fun_i];
    const auto &y_min = mat_data.y_min, &y_max = mat_data.y_max;
    for (int d = 0; d < dim; ++d) {
      auto new_Sig = Sig[d][d];
      y_c_f(new_Sig, y_min, y_max);
      p.Jp *= Sig[d][d] / new_Sig;
      Sig[d][d] = new_Sig, J *= new_Sig;
    }
    c_F[mat_fun_i](p.F, U, Sig, V);
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
      gvm[1] -= dt * 9.8 * 2.f;
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
}  // namespace mpmt
}  // namespace dym

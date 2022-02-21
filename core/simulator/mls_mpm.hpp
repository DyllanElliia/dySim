/*
 * @Author: DyllanElliia
 * @Date: 2022-02-10 15:49:14
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-17 17:52:38
 * @Description: mls-mpm based simulator.
 */
#pragma once
#include "../dyMath.hpp"
namespace dym {
namespace {
#define dim 3
typedef unsigned int u_int;
typedef unsigned short u_short;
#define cconst constexpr const
typedef dym::Matrix<Real, 3, 3> Matrix3;
typedef dym::Vector<Real, 3> Vector3;
typedef dym::Vector<Real, 4> Vector4;
typedef dym::Vector<int, 3> Vector3i;
#define identity3 dym::matrix::identity<Real, 3>()
Real dx, inv_dx;
u_int n_particles;
u_short bound;
u_short n_grid;
Vector3 globalForce;

Real p_rho = 1, p_vol = std::pow(dx * 0.5, 2), p_mass = p_vol * p_rho,
     hardening = 10.f;
Real E = 0.1e4f, nu = 0.2f;
Real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));
// struct material {
//   Real p_rho, p_vol, p_mass, hardening, mu_0, lambda_0;
//   material(const Real &E, const Real &nu, const Real &hardening,
//            const Real &p_rho, const Real &p_vol)
//       : mu_0(E / (2 * (1 + nu))),
//         lambda_0(E * nu / ((1 + nu) * (1 - 2 * nu))),
//         hardening(hardening),
//         p_rho(p_rho),
//         p_vol(p_vol),
//         p_mass(p_vol * p_rho) {}
// };
struct Particle_o {
  Vector3 v;
  Matrix3 C, F;
  Real Jp;
  unsigned short material;
  Particle_o() : v(0), C(0), F(identity3), Jp(1.f), material(0) {}
};

// std::vector<material> material_l;
std::vector<Particle_o> particles;

dym::Tensor<Vector4> grid_vm;

std::vector<std::function<void(Real &, Real &, Particle_o &)>> c_mu_la;
std::vector<std::function<void(Real &)>> yield_criteria;
std::vector<std::function<void(Matrix3 &, Matrix3 &, Matrix3 &, Matrix3 &)>>
    c_F;
dym::Tensor<Vector3> x;
}  // namespace

enum SimGridAcc {
  LowGrid = (u_short)32,
  MidGrid = (u_short)48,
  HighGrid = (u_short)64
};

void init(SimGridAcc gridNum = SimGridAcc::MidGrid) {
  n_particles = 0, n_grid = gridNum;
  dx = 1 / Real(n_grid), inv_dx = Real(n_grid), bound = n_grid / 16;
  grid_vm =
      dym::Tensor<Vector4>(0, dym::gi(n_grid + 1, n_grid + 1, n_grid + 1));
  x = dym::Tensor<Vector3>(0, dym::gi(0));
}

Vector3 addGlobalForce(const Vector3 &f) {
  globalForce += f;
  return globalForce;
}
Vector3 setGlobalForce(const Vector3 &f) {
  globalForce = f;
  return globalForce;
}
auto &getPos() { return x; }
auto getParticlesNum() { return n_particles; }

// u_int addMaterial(const Real &E = 0.1e4f, const Real &nu = 0.2f,
//                   const Real &hardening = 10.f, const Real &p_rho = 1.f,
//                   Real p_vol = -1) {
//   if (p_vol < 0) p_vol = dym::pow(dx * 0.5, 2);
//   material_l.push_back(material(E, nu, hardening, p_rho, p_vol));
//   return material_l.size() - 1;
// }
u_int addLiquidMateria(const u_int &material_index, const Real &mu_ = 0.f) {
  if (c_mu_la.size() != material_index ||
      yield_criteria.size() != material_index || c_F.size() != material_index)
    return -1;
  // material &ma = material_l[material_index];
  // Real hardening = ma.hardening, mu_0 = ma.mu_0, lambda_0 = ma.lambda_0;
  qprint(hardening, mu_0, lambda_0, material_index);
  c_mu_la.push_back([&](Real &mu, Real &lambda, Particle_o &p) {
    auto h = dym::exp(hardening * (1.0f - p.Jp));
    mu = mu_, lambda = lambda_0 * h;
  });
  yield_criteria.push_back([&](Real &new_Sig) {});
  c_F.push_back([&](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {});
  return material_index;
}
u_int addJellyMateria(const u_int &material_index, const Real &h_) {
  if (c_mu_la.size() != material_index - 1 ||
      yield_criteria.size() != material_index - 1 ||
      c_F.size() != material_index - 1)
    return -1;
  // material &ma = material_l[material_index];
  // Real hardening = ma.hardening, mu_0 = ma.mu_0, lambda_0 = ma.lambda_0;
  c_mu_la.push_back([&](Real &mu, Real &lambda, Particle_o &p) {
    auto h = h_;
    mu = mu_0 * h, lambda = lambda_0 * h;
  });
  yield_criteria.push_back([&](Real &new_Sig) {});
  c_F.push_back([&](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {});
  return material_index;
}
u_int addPlasticMateria(const u_int &material_index, const Real &yield_min,
                        const Real &yield_max) {
  if (c_mu_la.size() != material_index - 1 ||
      yield_criteria.size() != material_index - 1 ||
      c_F.size() != material_index - 1)
    return -1;
  // material &ma = material_l[material_index];
  // Real hardening = ma.hardening, mu_0 = ma.mu_0, lambda_0 = ma.lambda_0;
  c_mu_la.push_back([&](Real &mu, Real &lambda, Particle_o &p) {
    auto h = dym::exp(hardening * (1.0f - p.Jp));
    mu = mu_0 * h, lambda = lambda_0 * h;
  });
  yield_criteria.push_back([&](Real &new_Sig) {
    new_Sig = dym::clamp(new_Sig, yield_min, yield_max);
  });
  c_F.push_back([&](Matrix3 &F, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {
    F = U * Sig * V.transpose();
  });
  return material_index;
}

u_int addParticle(Tensor<Vector3> &newX, const u_int &material_index) {
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
  // qprint(particles.size(), particles[100].F, particles[100].material,
  //        material_l[material_index].p_vol,
  //        material_l[material_index].p_mass);
  // qprint(material_l.size(), c_mu_la.size(), yield_criteria.size(),
  // c_F.size());
  return par_num;
}

void advance(const Real &dt) {
  // qprint(grid_vm[dym::gi(24, 5, 24)]);
  grid_vm = Vector4(0.f);
  // qprint(grid_vm[dym::gi(24, 5, 24)]);

  x.for_each_i([&](Vector3 &px, int pi) {
    auto Xp = px / dx;
    auto base = (Xp - Vector3(0.5f)).cast<int>();
    Vector3 fx = Xp - base.cast<Real>();
    std::array<Vector3, 3> w = {Vector3(0.50f) * dym::sqr(Vector3(1.5f) - fx),
                                Vector3(0.75f) - dym::sqr(fx - Vector3(1.0f)),
                                Vector3(0.50f) * dym::sqr(fx - Vector3(0.5f))};
    auto &p = particles[pi];
    // auto &material_data = material_l[p.material];
    // auto &p_vol = material_data.p_vol, &p_mass = material_data.p_mass;
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
      gvm += Vector4(dt * globalForce);
      // gvm[1] -= dt * 9.8 * 2;
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
    // px[1] = dym::max(px[1], dx * bound);
    if (pi == 100) qprint(px, p.v, dx / dt, dx * bound);
  });
}

}  // namespace dym
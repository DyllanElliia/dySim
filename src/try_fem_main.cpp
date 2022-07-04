/*
 * @Author: DyllanElliia
 * @Date: 2022-06-20 17:03:28
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-07-04 16:12:55
 * @Description:
 */
#include <dyGraphic.hpp>
#include <dyMath.hpp>

const int N = 12;
const Real dt = 5e-5;
const Real dx = 1.0 / N;
const Real rho = 4e1;
const int NF = 2 * dym::exp(N);
const int NV = dym::pow(N + 1, 2);
const Real E = 4e4, nu = 0.2;
const Real mu = E / 2 / (1 + nu), lam = E * nu / (1 + nu) / (1 - 2 * nu);

dym::Vector2 ball_pos({0.2, 0.2});
Real ball_radius = 0.31;
Real damping = 14.5;

dym::Tensor<dym::Vector2> pos(0, dym::gi(NV));
dym::Tensor<dym::Vector2> vel(0, dym::gi(NV));
dym::Tensor<dym::Vector3i> f2v(0, dym::gi(NF));
dym::Tensor<dym::Matrix<Real, 2, 2>> B(0, dym::gi(NF)), F(0, dym::gi(NF));
dym::Tensor<Real> V(0, dym::gi(NF)), phi(0, dym::gi(NF));
Real U = 0;

dym::Vector2 gravity({0, 0}), attractor_pos({0, 0}), attractor_strength({0, 0});

_DYM_FORCE_INLINE_ void update_U() {
  F.for_each_i([&](dym::Matrix<Real, 2, 2> &Fi, int i) {
    const auto &ia = f2v[i][0], &ib = f2v[i][1], &ic = f2v[i][2];
    auto &a = pos[ia], &b = pos[ib], &c = pos[ic];
    V[i] = dym::abs(((a - c).cross(b - c)).length());
    dym::Matrix<Real, 2, 2> D_i;
    D_i.setColVec(0, a - c), D_i.setColVec(1, b - c);
    F[i] = D_i * B[i];
  });
  phi.for_each_i([&](Real &phi_it, int i) {
    auto F_i = F[i];
    auto log_J_i = dym::log(F_i.det());
    auto phi_i = mu / 2 * ((F_i.tr() - F_i).trace() - 2);
    phi_i -= mu * log_J_i + phi_i;
    phi_i += lam / 2 * dym::exp(log_J_i);
    phi_it = phi_i;
#pragma omp atomic
    U += V[i] * phi_i;
  });
}

_DYM_FORCE_INLINE_ auto solveGrad(int &i) {}

void advance() {}

template <typename type> type tryFun(type x) { return 3.0 * x * x + 2.0 * x; }

int main(int argc, char const *argv[]) {
  dym::GUI("fem", dym::gi(0, 0, 0));
  dym::Vector<int, 3> v{1, 2, 3};
  v.show();
  dym::Matrix<Real, 2, 2> m{{1.0, 2.0}, {3.0, 4.0}};
  m.show();
  Real x = 2.f;
  auto res = tryFun(dym::DualNum<Real>{x, 1});
  qprint(res);

  dym::Vector<dym::DualNum<Real>, 3> point{1.0, 2.0, 3.0};
  point.show();

  dym::Matrix<dym::DualNum<Real>, 3, 3> mat{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto resmp = mat * point;
  qprint(resmp);
  return 0;
}

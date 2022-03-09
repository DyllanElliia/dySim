/*
 * @Author: DyllanElliia
 * @Date: 2021-11-18 17:50:20
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-04 16:02:15
 * @Description:
 */
#pragma once
#include "../matALG.hpp"

namespace dym {
namespace traditionalSVD {
namespace {
#define GAMMA 5.8284271247
#define C_STAR 0.9238795325
#define S_STAR 0.3826834323
#define SVD_EPS 0.0000001
typedef Real real;
typedef Vector<Real, 2> Vector2;
typedef Vector<Real, 3> Vector3;
typedef Vector<Real, 4> Vector4;
typedef Matrix<Real, 3, 3> Matrix3;
typedef Matrix<Real, 4, 4> Matrix4;
typedef Matrix<Real, 3, 3> Matrix3;

template <std::size_t n>
inline Matrix<Real, n, n> transpose(const Matrix<Real, n, n> &m) {
  return m.transpose();
}
#define transposed transpose

inline Vector2 approx_givens_quat(real s_pp, real s_pq, real s_qq) {
  real c_h = 2 * (s_pp - s_qq);
  real s_h2 = s_pq * s_pq;
  real c_h2 = c_h * c_h;
  if (GAMMA * s_h2 < c_h2) {
    real omega = 1.0f / std::sqrt(s_h2 + c_h2);
    return Vector2({omega * c_h, omega * s_pq});
  }
  return Vector2({C_STAR, S_STAR});
}

// the quaternion is stored in vec4 like so:
// (c, s * vec3) meaning that .x() = c
inline Matrix3 quat_to_mat3(Vector4 quat) {
  real qx2 = quat.y() * quat.y();
  real qy2 = quat.z() * quat.z();
  real qz2 = quat.w() * quat.w();
  real qwqx = quat.x() * quat.y();
  real qwqy = quat.x() * quat.z();
  real qwqz = quat.x() * quat.w();
  real qxqy = quat.y() * quat.z();
  real qxqz = quat.y() * quat.w();
  real qyqz = quat.z() * quat.w();

  Matrix3 r;
  r[0][0] = 1.0f - 2.0f * (qy2 + qz2), r[1][0] = 2.0f * (qxqy + qwqz),
  r[2][0] = 2.0f * (qxqz - qwqy);
  r[0][1] = 2.0f * (qxqy - qwqz), r[1][1] = 1.0f - 2.0f * (qx2 + qz2),
  r[2][1] = 2.0f * (qyqz + qwqx);
  r[0][2] = 2.0f * (qxqz + qwqy), r[1][2] = 2.0f * (qyqz - qwqx),
  r[2][2] = 1.0f - 2.0f * (qx2 + qy2);
  return r;
}

Matrix3 symmetric_eigenanalysis(Matrix3 A) {
  Matrix3 S = transpose(A) * A;
  // jacobi iteration
  Matrix3 q([](real &e, int i, int j) { e = i == j ? 1.f : 0; });
  for (int i = 0; i < 5; i++) {
    Vector2 ch_sh = approx_givens_quat(S.getColVec(0).x(), S.getColVec(0).y(),
                                       S.getColVec(1).y());
    Vector4 ch_sh_quat = Vector4({ch_sh.x(), 0, 0, ch_sh.y()});
    Matrix3 q_mat = quat_to_mat3(ch_sh_quat);
    S = transpose(q_mat) * S * q_mat;
    q = q * q_mat;

    ch_sh = approx_givens_quat(S.getColVec(0).x(), S.getColVec(0).z(),
                               S.getColVec(2).z());
    ch_sh_quat = Vector4({ch_sh.x(), 0, -ch_sh.y(), 0});
    q_mat = quat_to_mat3(ch_sh_quat);
    S = transpose(q_mat) * S * q_mat;
    q = q * q_mat;

    ch_sh = approx_givens_quat(S.getColVec(1).y(), S.getColVec(1).z(),
                               S.getColVec(2).z());
    ch_sh_quat = Vector4({ch_sh.x(), ch_sh.y(), 0, 0});
    q_mat = quat_to_mat3(ch_sh_quat);
    S = transpose(q_mat) * S * q_mat;
    q = q * q_mat;
  }
  return q;
}

inline Vector2 approx_qr_givens_quat(real a0, real a1) {
  real rho = std::sqrt(a0 * a0 + a1 * a1);
  real s_h = a1;
  real max_rho_eps = rho;
  if (rho <= SVD_EPS) {
    s_h = 0;
    max_rho_eps = SVD_EPS;
  }
  real c_h = max_rho_eps + a0;
  if (a0 < 0) {
    real temp = c_h - 2 * a0;
    c_h = s_h;
    s_h = temp;
  }
  real omega = 1.0f / std::sqrt(c_h * c_h + s_h * s_h);
  return Vector2({omega * c_h, omega * s_h});
}

struct QR_mats {
  Matrix3 Q;
  Matrix3 R;
};

inline QR_mats qr_decomp(Matrix3 B) {
  QR_mats qr_decomp_result;
  Matrix3 R;
  // 1 0
  // (ch, 0, 0, sh)
  Vector2 ch_sh10 =
      approx_qr_givens_quat(B.getColVec(0).x(), B.getColVec(0).y());
  Matrix3 Q10 = quat_to_mat3(Vector4({ch_sh10.x(), 0, 0, ch_sh10.y()}));
  R = transpose(Q10) * B;

  // 2 0
  // (ch, 0, -sh, 0)
  Vector2 ch_sh20 =
      approx_qr_givens_quat(R.getColVec(0).x(), R.getColVec(0).z());
  Matrix3 Q20 = quat_to_mat3(Vector4({ch_sh20.x(), 0, -ch_sh20.y(), 0}));
  R = transpose(Q20) * R;

  // 2 1
  // (ch, sh, 0, 0)
  Vector2 ch_sh21 =
      approx_qr_givens_quat(R.getColVec(1).y(), R.getColVec(1).z());
  Matrix3 Q21 = quat_to_mat3(Vector4({ch_sh21.x(), ch_sh21.y(), 0, 0}));
  R = transpose(Q21) * R;

  qr_decomp_result.R = R;

  qr_decomp_result.Q = Q10 * Q20 * Q21;
  return qr_decomp_result;
}
}  // namespace

inline void svd(Matrix3 A, Matrix3 &U, Matrix3 &Sig, Matrix3 &V) {
  V = symmetric_eigenanalysis(A);

  Matrix3 B = A * V;
  // sort singular values
  real rho0 = vector::dot(B.getColVec(0), B.getColVec(0));
  real rho1 = vector::dot(B.getColVec(1), B.getColVec(1));
  real rho2 = vector::dot(B.getColVec(2), B.getColVec(2));
  if (rho0 < rho1) {
    Vector3 temp = B.getColVec(1);
    B.setColVec(1, -B.getColVec(0));
    B.setColVec(0, temp);
    temp = V.getColVec(1);
    V.setColVec(1, -V.getColVec(0));
    V.setColVec(0, temp);
    real temp_rho = rho0;
    rho0 = rho1;
    rho1 = temp_rho;
  }
  if (rho0 < rho2) {
    Vector3 temp = B.getColVec(2);
    B.setColVec(2, -B.getColVec(0));
    B.setColVec(0, temp);
    temp = V.getColVec(2);
    V.setColVec(2, -V.getColVec(0));
    V.setColVec(0, temp);
    rho2 = rho0;
  }
  if (rho1 < rho2) {
    Vector3 temp = B.getColVec(2);
    B.setColVec(2, -B.getColVec(1));
    B.setColVec(1, temp);
    temp = V.getColVec(2);
    V.setColVec(2, -V.getColVec(1));
    V.setColVec(1, temp);
  }

  QR_mats QR = qr_decomp(B);
  U = QR.Q;
  Sig = QR.R;
}

inline void SVD_to_polar(Matrix3 &u, Matrix3 &s, Matrix3 &v, Matrix3 &U,
                         Matrix3 &P) {
  P = v * s * transposed(v);
  U = u * transposed(v);
}

inline void polar_decomp(Matrix3 A, Matrix3 &U, Matrix3 &P) {
  Matrix3 u, s, v;
  svd(A, u, s, v);
  P = v * s * transposed(v);
  U = u * transposed(v);
}

}  // namespace traditionalSVD

}  // namespace dym
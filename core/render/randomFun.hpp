/*
 * @Author: DyllanElliia
 * @Date: 2022-03-07 16:56:57
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-23 15:29:47
 * @Description:
 */
#pragma once
#include <random>

#include "../dyMath.hpp"
#include "math/define.hpp"

namespace dym {
namespace rt {
typedef Vector3 Point3;
typedef Vector3 ColorRGB;
typedef Vector4 ColorRGBA;
using std::make_shared;
using std::shared_ptr;

// Constants

const Real infinity = std::numeric_limits<Real>::infinity();
const Real pi = 3.1415926535897932385;

// Utility Functions

_DYM_FORCE_INLINE_ Real degrees_to_radians(const Real &degrees) {
  return degrees * pi / 180.0;
}

namespace {} // namespace

_DYM_FORCE_INLINE_ Real random_real(const Real &min = 0.0,
                                    const Real &max = 1.0) {
  std::uniform_real_distribution<Real> distribution(min, max);
  static std::mt19937 generator;
  // return (distribution(generator) * (max - min)) - min;
  return distribution(generator);
}

_DYM_FORCE_INLINE_ Vector3 vec_random(const Real &min = 0.0,
                                      const Real &max = 1.0) {
  return Vector3(
      {random_real(min, max), random_real(min, max), random_real(min, max)});
}

_DYM_AUTO_INLINE_ Vector3 random_in_unit_sphere() {
  while (true) {
    auto p = vec_random(-1, 1);
    if (p.length_sqr() < 1)
      return p;
  }
}

_DYM_FORCE_INLINE_ Vector3 random_unit_vector() {
  return random_in_unit_sphere().normalize();
}

_DYM_FORCE_INLINE_ Vector3 random_in_hemisphere(const Vector3 &normal) {
  Vector3 in_unit_sphere = random_in_unit_sphere();
  if (vector::dot(in_unit_sphere, normal) >
      0.f) // In the same hemisphere as the normal
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}

_DYM_AUTO_INLINE_ Vector3 random_in_unit_disk() {
  while (true) {
    auto p = Vector3({random_real(-1, 1), random_real(-1, 1), 0});
    if (p.length_sqr() >= 1)
      continue;
    return p;
  }
}

_DYM_FORCE_INLINE_ Vector3 random_cosine_direction() {
  auto r1 = random_real();
  auto r2 = random_real();
  auto z = sqrt(1 - r2);

  auto phi = 2 * pi * r1;
  auto x = cos(phi) * sqrt(r2);
  auto y = sin(phi) * sqrt(r2);

  return Vector3({x, y, z});
}

_DYM_FORCE_INLINE_ Real solve_cosine_pdf(const Real &NdotL) {
  return (NdotL <= 0) ? 0 : NdotL / pi;
}

_DYM_FORCE_INLINE_ Vector3 random_GTR1_direction(const Real &xi_1,
                                                 const Real &xi_2,
                                                 const Real &alpha) {
  Real phi_h = 2.0 * pi * xi_1;
  Real sin_phi_h = sin(phi_h);
  Real cos_phi_h = cos(phi_h);

  Real cos_theta_h =
      sqrt((1.0 - pow(alpha * alpha, 1.0 - xi_2)) / (1 - alpha * alpha));
  Real sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));
  return Vector3(
      {sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h});
}

_DYM_FORCE_INLINE_ Real solve_GTR1_pdf(const Real &NdotH, const Real &alpha) {
  if (alpha >= 1.)
    return 1 / pi;
  Real a2 = alpha * alpha;
  Real t = 1 + (a2 - 1) * NdotH * NdotH;
  return (a2 - 1) / (pi * log(a2) * t);
}

_DYM_FORCE_INLINE_ Vector3 random_GTR2_direction(const Real &xi_1,
                                                 const Real &xi_2,
                                                 const Real &alpha) {
  Real phi_h = 2.0 * pi * xi_1;
  Real sin_phi_h = sin(phi_h);
  Real cos_phi_h = cos(phi_h);

  Real cos_theta_h = sqrt((1.0 - xi_2) / (1.0 + (alpha * alpha - 1.0) * xi_2));
  Real sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));
  return Vector3(
      {sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h});
}

_DYM_FORCE_INLINE_ Real solve_GTR2_pdf(const Real &NdotH, const Real &alpha) {
  Real a2 = alpha * alpha;
  Real t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
  return a2 / (pi * t * t);
}

_DYM_FORCE_INLINE_ Real solve_GTR2_aniso_pdf(const Real &dotHX,
                                             const Real &dotHY,
                                             const Real &dotNH, const Real &ax,
                                             const Real &ay) {
  Real deno =
      dotHX * dotHX / (ax * ax) + dotHY * dotHY / (ay * ay) + dotNH * dotNH;
  return 1.0 / (pi * ax * ay * deno * deno);
}

_DYM_FORCE_INLINE_ Real SchlickFresnel(const Real &u) {
  Real m = clamp(1 - u, 0, 1);
  Real m2 = m * m;
  return m2 * m2 * m; // pow(m,5)
}

// _DYM_FORCE_INLINE_ Real smithG_GGX(const Real &NdotV, const Real &alphaG) {
//   Real a = alphaG * alphaG;
//   Real b = NdotV * NdotV;
//   return NdotV / (NdotV + sqrt(a + b - a * b));
// }

_DYM_FORCE_INLINE_ Real smithG_GGX(const Real &NdotV, const Real &alphaG) {
  Real a = alphaG * alphaG;
  Real b = NdotV * NdotV;
  return 2. / (1. + sqrt(a + b - a * b));
}

_DYM_FORCE_INLINE_ Real smithG_GGX_aniso(const Real &NdotV, const Real &dotVX,
                                         const Real &dotVY, const Real &ax,
                                         const Real &ay) {
  return 2 * NdotV /
         (NdotV +
          sqrt(pow(dotVX * ax, 2.0) + pow(dotVY * ay, 2.0) + pow(NdotV, 2.0)));
}

} // namespace rt
} // namespace dym
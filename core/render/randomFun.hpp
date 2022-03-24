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

namespace dym {
namespace rt {
typedef Vector3 Point3;
typedef Vector3 ColorRGB;
using std::make_shared;
using std::shared_ptr;

// Constants

const Real infinity = std::numeric_limits<Real>::infinity();
const Real pi = 3.1415926535897932385;

// Utility Functions

_DYM_FORCE_INLINE_ Real degrees_to_radians(const Real& degrees) {
  return degrees * pi / 180.0;
}

namespace {}  // namespace

_DYM_FORCE_INLINE_ Real random_real(const Real& min = 0.0,
                                    const Real& max = 1.0) {
  std::uniform_real_distribution<Real> distribution(min, max);
  static std::mt19937 generator;
  // return (distribution(generator) * (max - min)) - min;
  return distribution(generator);
}

_DYM_FORCE_INLINE_ Vector3 vec_random(const Real& min = 0.0,
                                      const Real& max = 1.0) {
  return Vector3(
      {random_real(min, max), random_real(min, max), random_real(min, max)});
}

Vector3 random_in_unit_sphere() {
  while (true) {
    auto p = vec_random(-1, 1);
    if (p.length_sqr() < 1) return p;
  }
}

_DYM_FORCE_INLINE_ Vector3 random_unit_vector() {
  return random_in_unit_sphere().normalize();
}

_DYM_FORCE_INLINE_ Vector3 random_in_hemisphere(const Vector3& normal) {
  Vector3 in_unit_sphere = random_in_unit_sphere();
  if (vector::dot(in_unit_sphere, normal) >
      0.f)  // In the same hemisphere as the normal
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}

Vector3 random_in_unit_disk() {
  while (true) {
    auto p = Vector3({random_real(-1, 1), random_real(-1, 1), 0});
    if (p.length_sqr() >= 1) continue;
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
}  // namespace rt
}  // namespace dym
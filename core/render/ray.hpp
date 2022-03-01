/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:50:58
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-01 15:33:29
 * @Description:
 */
#pragma once
#include "../dyMath.hpp"
#include <random>

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

_DYM_FORCE_INLINE_ Real random_real(const Real& min = 0.f,
                                    const Real& max = 1.f) {
  static std::uniform_real_distribution<Real> distribution(min, max);
  static std::mt19937 generator;
  return distribution(generator);
}

_DYM_FORCE_INLINE_ Vector3 vec_random(const Real& min, const Real& max) {
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

class Ray {
 public:
  Ray() {}
  Ray(const Point3& origin, const Vector3& direction)
      : orig(origin), dir(direction) {}

  Point3 origin() const { return orig; }
  Vector3 direction() const { return dir; }

  Point3 at(Real t) const { return orig + t * dir; }

 public:
  Point3 orig;  // original point
  Vector3 dir;  // direction vector
};

struct HitRecord {
  Point3 p;
  Vector3 normal;
  Real t;
  bool front_face;

  inline void set_face_normal(const Ray& r, const Vector3& outward_normal) {
    front_face = vector::dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

}  // namespace rt
}  // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:50:58
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-18 18:10:34
 * @Description:
 */
#pragma once
#include "randomFun.hpp"

#include <any>

namespace dym {
namespace rt {
class Ray {
public:
  Ray() {}
  Ray(const Point3 &origin, const Vector3 &direction, const Real &time = 0.f)
      : orig(origin), dir(direction), tm(time) {}

  Point3 origin() const { return orig; }
  Vector3 direction() const { return dir; }
  Real time() const { return tm; }

  Point3 at(Real t) const { return orig + t * dir; }

public:
  Point3 orig; // original point
  Vector3 dir; // direction vector
  Real tm;
};

class Material;
class pdf;
struct HitRecord {
  Point3 p;
  Vector3 normal;
  Real t;
  Real u, v;
  bool front_face;
  shared_ptr<Material> mat_ptr;
  int obj_id;

  inline void set_face_normal(const Ray &r, const Vector3 &outward_normal) {
    front_face = vector::dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class Photon {
public:
  Ray r;
  shared_ptr<Material> mat_ptr;
  Photon(const Ray &r, shared_ptr<Material> mat_ptr) : r(r), mat_ptr(mat_ptr) {}
  Photon() {}
  ~Photon() {}
};

struct ScatterRecord {
  Ray specular_ray;
  Real is_specular;
  ColorRGB attenuation;
  shared_ptr<pdf> pdf_ptr;
  std::any otherData;
};

} // namespace rt
} // namespace dym
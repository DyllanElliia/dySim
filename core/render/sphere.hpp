/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:00:10
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-01 15:06:18
 * @Description:
 */
#pragma once

#include "baseClass.hpp"

namespace dym {
namespace rt {
class Sphere : public Hittable {
 public:
  Sphere() {}
  Sphere(Point3 cen, Real r) : center(cen), radius(r){};

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;

 public:
  Point3 center;
  Real radius;
};

bool Sphere::hit(const Ray& r, Real t_min, Real t_max, HitRecord& rec) const {
  Vector3 oc = r.origin() - center;
  auto a = r.direction().length_sqr();
  auto half_b = vector::dot(oc, r.direction());
  auto c = oc.length_sqr() - radius * radius;

  auto discriminant = half_b * half_b - a * c;
  if (discriminant < 0) return false;
  auto sqrtd = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  auto root = (-half_b - sqrtd) / a;
  if (root < t_min || t_max < root) {
    root = (-half_b + sqrtd) / a;
    if (root < t_min || t_max < root) return false;
  }

  rec.t = root;
  rec.p = r.at(rec.t);
  Vector3 outward_normal = (rec.p - center) / radius;
  rec.set_face_normal(r, outward_normal);
  return true;
}
}  // namespace rt
}  // namespace dym
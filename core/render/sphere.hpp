/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:00:10
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-03 15:21:46
 * @Description:
 */
#pragma once

#include "baseClass.hpp"

namespace dym {
namespace rt {
class Sphere : public Hittable {
 public:
  Sphere() {}
  Sphere(Point3 cen, Real r, shared_ptr<Material> m)
      : center(cen), radius(r), mat_ptr(m){};

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;

 public:
  Point3 center;
  Real radius;
  shared_ptr<Material> mat_ptr;
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
  rec.mat_ptr = mat_ptr;
  return true;
}
}  // namespace rt
}  // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:00:10
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-04 14:55:00
 * @Description:
 */
#pragma once

#include "baseClass.hpp"

namespace dym {
namespace rt {
class Sphere : public Hittable {
 private:
  static void get_sphere_uv(const Point3& p, Real& u, Real& v) {
    // p: a given point on the sphere of radius one, centered at the origin.
    // u: returned value [0,1] of angle around the Y axis from X=-1.
    // v: returned value [0,1] of angle from Y=-1 to Y=+1.
    //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
    //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
    //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + pi;

    u = phi / (2 * pi);
    v = theta / pi;
  }

 public:
  Sphere() {}
  Sphere(Point3 cen, Real r, shared_ptr<Material> m)
      : center(cen), radius(r), mat_ptr(m){};

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;
  virtual bool bounding_box(aabb& output_box) const override;

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
  get_sphere_uv(outward_normal, rec.u, rec.v);
  rec.mat_ptr = mat_ptr;
  return true;
}
bool Sphere::bounding_box(aabb& output_box) const {
  output_box = aabb(center - radius, center + radius);
  return true;
}
}  // namespace rt
}  // namespace dym
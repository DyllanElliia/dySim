/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:00:10
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-18 18:12:21
 * @Description:
 */
#pragma once

#include "../baseClass.hpp"

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
    if (isnan(u)) u = 0;
    if (isnan(v)) v = p.y() > 1.f ? 1 : 0;
  }

 public:
  Sphere() {}
  Sphere(Point3 cen, Real r, shared_ptr<Material> m)
      : center(cen), radius(r), mat_ptr(m){};

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;
  virtual bool bounding_box(aabb& output_box) const override;

  virtual Real pdf_value(const Point3& origin, const Vector3& v) const override;
  virtual Vector3 random(const Point3& origin) const override;

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
  rec.obj_id = (int)this;
  return true;
}
bool Sphere::bounding_box(aabb& output_box) const {
  output_box = aabb(center - radius, center + radius);
  return true;
}

namespace {
_DYM_FORCE_INLINE_ Vector3 random_to_sphere(const Real& radius,
                                            const Real& distance_squared) {
  auto r1 = random_real();
  auto r2 = random_real();
  auto z = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);

  auto phi = 2 * pi * r1;
  auto x = cos(phi) * sqrt(1 - z * z);
  auto y = sin(phi) * sqrt(1 - z * z);

  return Vector3({x, y, z});
}
}  // namespace

Real Sphere::pdf_value(const Point3& o, const Vector3& v) const {
  HitRecord rec;
  if (!this->hit(Ray(o, v), 0.001, infinity, rec)) return 0;

  auto cos_theta_max = sqrt(1 - radius * radius / (center - o).length_sqr());
  auto solid_angle = 2 * pi * (1 - cos_theta_max);

  return 1 / solid_angle;
}

Vector3 Sphere::random(const Point3& o) const {
  Vector3 direction = center - o;
  auto distance_squared = direction.length_sqr();
  onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(random_to_sphere(radius, distance_squared));
}
}  // namespace rt
}  // namespace dym
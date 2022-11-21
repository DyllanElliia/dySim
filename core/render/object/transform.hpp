/*
 * @Author: DyllanElliia
 * @Date: 2022-03-11 14:57:05
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-06-20 17:47:13
 * @Description:
 */
#pragma once

#include "../baseClass.hpp"

namespace dym {
namespace rt {
class Transform3 : public Hittable {
public:
  Transform3(const shared_ptr<Hittable> &ptr, const Matrix3 &mat,
             const Vector3 &offset = 0)
      : ptr(ptr), mat(mat), offset(offset) {
    mat_inv = mat.inverse();
    mat_norm_it = mat.inverse().transpose();
    hasbox = ptr->bounding_box(bbox);

    const auto infinity = std::numeric_limits<Real>::infinity();
    Point3 minp(infinity), maxp(-infinity);
    const auto &bmin = bbox.min(), &bmax = bbox.max();

    Loop<int, 2>([&](auto i) {
      Loop<int, 2>([&](auto j) {
        Loop<int, 2>([&](auto k) {
          Point3 tester({i * bmax.x() + (1 - i) * bmin.x(),
                         j * bmax.y() + (1 - j) * bmin.y(),
                         k * bmax.z() + (1 - k) * bmin.z()});
          tester = mat * tester;
          minp = min(minp, tester), maxp = max(maxp, tester);
        });
      });
    });
    bbox = aabb(minp + offset, maxp + offset);
  }

  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const override;
  virtual bool bounding_box(aabb &output_box) const override;

public:
  shared_ptr<Hittable> ptr;
  Matrix3 mat, mat_inv, mat_norm_it;
  Vector3 offset;
  bool hasbox;
  aabb bbox;
};

bool Transform3::hit(const Ray &r, Real t_min, Real t_max,
                     HitRecord &rec) const {
  auto origin = mat_inv * (r.origin() - offset);
  auto direction = (mat_inv * r.direction()).normalize();
  Ray tf_r(origin, direction, r.time());

  if (!ptr->hit(tf_r, 1e-6, infinity, rec))
    return false;

  rec.p = mat * rec.p + offset;
  rec.normal = (mat_norm_it * rec.normal).normalize();
  rec.t = (rec.p - r.origin())[0] / r.direction()[0];

  return (rec.t > t_min && rec.t < t_max);
}
bool Transform3::bounding_box(aabb &output_box) const {
  output_box = bbox;
  return hasbox;
}

} // namespace rt
} // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2022-04-11 14:22:27
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-11 14:31:29
 * @Description:
 */
#pragma once

#include "triangle.hpp"

namespace dym {
namespace rt {

class Mesh : public Hittable {
 public:
  Mesh() {}
  Mesh(Point3 cen, Real r, shared_ptr<Material> m);

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;
  virtual bool bounding_box(aabb& output_box) const override;

  virtual Real pdf_value(const Point3& origin, const Vector3& v) const override;
  virtual Vector3 random(const Point3& origin) const override;
};
}  // namespace rt
}  // namespace dym
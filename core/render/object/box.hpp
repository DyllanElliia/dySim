/*
 * @Author: DyllanElliia
 * @Date: 2022-03-07 17:08:29
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-24 14:45:03
 * @Description:
 */
#pragma once
#include "../BVH/bvhNode.hpp"
#include "aaRect.hpp"
namespace dym {
namespace rt {
class Box : public Hittable {
 public:
  Box() {}
  Box(const Point3& pmin, const Point3& pax, shared_ptr<Material> ptr);

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;

  virtual bool bounding_box(aabb& output_box) const override {
    output_box = aabb(box_min, box_max);
    return true;
  }

  // virtual Real pdf_value(const Point3& origin, const Vector3& v) const
  // override; virtual Vector3 random(const Point3& origin) const override;

 public:
  Point3 box_min;
  Point3 box_max;
  BvhNode sides;
};

Box::Box(const Point3& p0, const Point3& p1, shared_ptr<Material> ptr) {
  box_min = p0;
  box_max = p1;

  HittableList side;

  side.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
  side.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

  side.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
  side.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

  side.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
  side.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));

  sides = BvhNode(side);
}

bool Box::hit(const Ray& r, Real t_min, Real t_max, HitRecord& rec) const {
  return sides.hit(r, t_min, t_max, rec);
}

}  // namespace rt
}  // namespace dym
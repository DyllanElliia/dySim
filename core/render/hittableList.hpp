/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:07:40
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-04 14:59:17
 * @Description:
 */
#pragma once
#include "baseClass.hpp"
namespace dym {
namespace rt {
class HittableList : public Hittable {
 public:
  HittableList() {}
  HittableList(shared_ptr<Hittable> object) { add(object); }

  void clear() { objects.clear(); }
  void add(shared_ptr<Hittable> object) { objects.push_back(object); }

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;
  virtual bool bounding_box(aabb& output_box) const override;

 public:
  std::vector<shared_ptr<Hittable>> objects;
};

bool HittableList::hit(const Ray& r, Real t_min, Real t_max,
                       HitRecord& rec) const {
  HitRecord temp_rec;
  bool hit_anything = false;
  auto closest_so_far = t_max;

  for (const auto& object : objects) {
    if (object->hit(r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }

  return hit_anything;
}
bool HittableList::bounding_box(aabb& output_box) const {
  if (objects.empty()) return false;

  aabb temp_box;
  bool first_box = true;

  for (const auto& object : objects) {
    if (!object->bounding_box(temp_box)) return false;
    output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
    first_box = false;
  }

  return true;
}
}  // namespace rt
}  // namespace dym
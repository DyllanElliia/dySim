/*
 * @Author: DyllanElliia
 * @Date: 2022-03-04 14:59:47
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-18 15:17:15
 * @Description:
 */
#pragma once
#include "../hittableList.hpp"
#include <algorithm>
#include <execution>
namespace dym {
namespace rt {
class BvhNode : public Hittable {
 public:
  BvhNode() {}

  BvhNode(HittableList& list) : BvhNode(list.objects, 0, list.objects.size()) {}

  BvhNode(std::vector<shared_ptr<Hittable>>& src_objects, size_t start,
          size_t end, const Real& isrlevel = 1);

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;

  virtual bool bounding_box(aabb& output_box) const override;

 public:
  shared_ptr<Hittable> left;
  shared_ptr<Hittable> right;
  aabb box;
};

bool BvhNode::hit(const Ray& r, Real t_min, Real t_max, HitRecord& rec) const {
  if (!box.hit(r, t_min, t_max)) return false;
  // qprint("in BVH");
  bool hit_left = left->hit(r, t_min, t_max, rec);
  bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

  return hit_left || hit_right;
}

bool BvhNode::bounding_box(aabb& output_box) const {
  output_box = box;
  return true;
}
namespace {
inline bool box_compare(const shared_ptr<Hittable>& a,
                        const shared_ptr<Hittable>& b, const int& axis) {
  aabb box_a;
  aabb box_b;

  if (!a->bounding_box(box_a) || !b->bounding_box(box_b))
    std::cerr << "No bounding box in bvh_node constructor.\n";

  return box_a.min()[axis] < box_b.min()[axis];
}

_DYM_FORCE_INLINE_ bool box_x_compare(const shared_ptr<Hittable>& a,
                                      const shared_ptr<Hittable>& b) {
  return box_compare(a, b, 0);
}

_DYM_FORCE_INLINE_ bool box_y_compare(const shared_ptr<Hittable>& a,
                                      const shared_ptr<Hittable>& b) {
  return box_compare(a, b, 1);
}

_DYM_FORCE_INLINE_ bool box_z_compare(const shared_ptr<Hittable>& a,
                                      const shared_ptr<Hittable>& b) {
  return box_compare(a, b, 2);
}

}  // namespace

BvhNode::BvhNode(std::vector<shared_ptr<Hittable>>& src_objects, size_t start,
                 size_t end, const Real& isrlevel) {
  auto objects =
      src_objects;  // Create a modifiable array of the source scene objects

  int axis = random_real(0, 2);
  auto comparator = (axis == 0)   ? box_x_compare
                    : (axis == 1) ? box_y_compare
                                  : box_z_compare;

  size_t object_span = end - start;

  if (object_span == 1) {
    left = right = objects[start];
  } else if (object_span == 2) {
    if (comparator(objects[start], objects[start + 1])) {
      left = objects[start];
      right = objects[start + 1];
    } else {
      left = objects[start + 1];
      right = objects[start];
    }
  } else {
    // qprint("use BVH");
    if (random_real() < isrlevel)
      std::sort(std::execution::par, objects.begin() + start,
                objects.begin() + end, comparator);

    auto mid = start + object_span / 2;
    const Real nIsRLevel = 0.5;
    left = make_shared<BvhNode>(objects, start, mid, nIsRLevel);
    right = make_shared<BvhNode>(objects, mid, end, nIsRLevel);
  }

  aabb box_left, box_right;

  if (!left->bounding_box(box_left) || !right->bounding_box(box_right))
    std::cerr << "No bounding box in BvhNode constructor.\n";

  box = surrounding_box(box_left, box_right);
}
}  // namespace rt
}  // namespace dym
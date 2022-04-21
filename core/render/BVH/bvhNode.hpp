/*
 * @Author: DyllanElliia
 * @Date: 2022-03-04 14:59:47
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-21 17:26:05
 * @Description:
 */
#pragma once
#include "../hittableList.hpp"
#include <algorithm>
// #include <execution>
namespace dym {
namespace rt {
namespace {
bool useParallel = true;
}
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
      std::sort(objects.begin() + start, objects.begin() + end, comparator);
    // std::sort(std::execution::par, objects.begin() + start,
    //           objects.begin() + end, comparator);

    auto mid = start + object_span / 2;
    const Real nIsRLevel = 0.5;
    if (useParallel && object_span > 48) {
      useParallel = false;
      std::vector<std::shared_ptr<BvhNode>> bvhNode_subList(32);
      const int delStep = object_span / 32;
#pragma omp parallel for
      for (int ii = 0; ii < 32; ++ii) {
        int bi = start + ii * delStep, ei = start + (ii + 1) * delStep;
        if (ei > end) ei = end;
        bvhNode_subList[ii] = make_shared<BvhNode>(objects, bi, ei, 0.8);
      }
      for (int ii = 16; ii > 0; ii /= 2) {
        for (int jj = 0; jj < ii; ++jj) {
          std::shared_ptr<BvhNode> it = make_shared<BvhNode>();
          int firstOne = jj << 1, secondOnde = (jj << 1) + 1;
          // qprint("(", firstOne, ",", secondOnde, ")->", jj);
          it->left = bvhNode_subList[firstOne],
          it->right = bvhNode_subList[secondOnde];
          aabb box_left, box_right;
          if (!(it->left)->bounding_box(box_left) ||
              !(it->right)->bounding_box(box_right))
            std::cerr << "No bounding box in BvhNode constructor.\n";

          it->box = surrounding_box(box_left, box_right);
          bvhNode_subList[jj] = it;
        }
      }
      auto& copyNode = bvhNode_subList[0];
      left = copyNode->left;
      right = copyNode->right;
      box = copyNode->box;
      useParallel = true;
    } else {
      left = make_shared<BvhNode>(objects, start, mid, nIsRLevel);
      right = make_shared<BvhNode>(objects, mid, end, nIsRLevel);
    }
  }

  aabb box_left, box_right;

  if (!left->bounding_box(box_left) || !right->bounding_box(box_right))
    std::cerr << "No bounding box in BvhNode constructor.\n";

  box = surrounding_box(box_left, box_right);
}
}  // namespace rt
}  // namespace dym
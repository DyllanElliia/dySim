#pragma once
#include "../BVH/bvhNode.hpp"
#include "./aaRect.hpp"

namespace dym {
namespace rt {
class Skybox : public Hittable {
public:
  Skybox() {}
  Skybox(const std::vector<shared_ptr<Material>> &mat_ptrs) {
    if (mat_ptrs.size() < 6) {
      DYM_WARNING_cs("SkyBox", "Skybox need 6 picture path for generate.");
      return;
    }
    box_min = -1e7;
    box_max = 1e7;
    auto &p0 = box_min, &p1 = box_max;
    HittableList side;

    side.add(make_shared<xy_rect<true>>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(),
                                        mat_ptrs[4]));
    side.add(make_shared<xy_rect<false>>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(),
                                         mat_ptrs[5]));

    side.add(make_shared<xz_rect<false>>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(),
                                         mat_ptrs[2]));
    side.add(make_shared<xz_rect<true>>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(),
                                        mat_ptrs[3]));

    side.add(make_shared<yz_rect<false>>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(),
                                         mat_ptrs[1]));
    side.add(make_shared<yz_rect<true>>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(),
                                        mat_ptrs[0]));

    sides = BvhNode(side);
  }
  ~Skybox() {}

  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const override {
    Ray ro = r;
    ro.orig = 0.0;
    return sides.hit(ro, t_min, t_max, rec);
  }
  virtual bool bounding_box(aabb &output_box) const override {
    output_box = aabb(box_min, box_max);
    return true;
  }

  _DYM_FORCE_INLINE_ ColorRGB sample(const Ray &r) {
    HitRecord rec;
    // if (!hit(ro, 1e-7, infinity, rec))
    //   return 0.0;
    hit(r, 1e-7, infinity, rec);
    auto &hitMat = *(rec.mat_ptr);
    ColorRGB Le = rec.mat_ptr->emitted(r, rec);

    return Le;
  }

private:
  // shared_ptr<Material> mat_ptrs[6];
  Point3 box_min;
  Point3 box_max;
  BvhNode sides;
};

} // namespace rt
} // namespace dym
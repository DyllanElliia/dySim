/*
 * @Author: DyllanElliia
 * @Date: 2022-03-04 14:37:17
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-04 14:58:48
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
namespace dym {
namespace rt {
class aabb {
 public:
  aabb() {}
  aabb(const Point3& a, const Point3& b) {
    minimum = a;
    maximum = b;
  }

  Point3 min() const { return minimum; }
  Point3 max() const { return maximum; }

  bool hit(const Ray& r, Real t_min, Real t_max) const {
    // for (int a = 0; a < 3; a++) {
    //   auto t0 = fmin((minimum[a] - r.origin()[a]) / r.direction()[a],
    //                  (maximum[a] - r.origin()[a]) / r.direction()[a]);
    //   auto t1 = fmax((minimum[a] - r.origin()[a]) / r.direction()[a],
    //                  (maximum[a] - r.origin()[a]) / r.direction()[a]);
    //   t_min = fmax(t0, t_min);
    //   t_max = fmin(t1, t_max);
    //   if (t_max <= t_min) return false;
    // }
    // return true;
    Loop<int, 3>([&](auto a) {
      auto invD = 1.0f / r.direction()[a];
      auto t0 = (min()[a] - r.origin()[a]) * invD;
      auto t1 = (max()[a] - r.origin()[a]) * invD;
      if (invD < 0.0f) std::swap(t0, t1);
      t_min = t0 > t_min ? t0 : t_min;
      t_max = t1 < t_max ? t1 : t_max;
      if (t_max <= t_min) return false;
    });
    return true;
  }

  Point3 minimum;
  Point3 maximum;
};

_DYM_FORCE_INLINE_ aabb surrounding_box(const aabb& box0, const aabb& box1) {
  return aabb(min(box0.min(), box1.min()), max(box0.max(), box1.max()));
}
}  // namespace rt
}  // namespace dym
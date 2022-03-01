/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:17:32
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-01 17:00:46
 * @Description:
 */
#pragma once
#include "ray.hpp"

namespace dym {
namespace rt {
class Camera {
 public:
  Camera() {
    auto aspect_ratio = 16.f / 9.f;
    auto viewport_height = 2.f;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.f;

    origin = Point3(0.f);
    horizontal = Vector3({viewport_width, 0.f, 0.f});
    vertical = Vector3({0.f, viewport_height, 0.f});
    lower_left_corner = origin - horizontal / 2.f - vertical / 2.f -
                        Vector3({0.f, 0.f, focal_length});
  }

  _DYM_FORCE_INLINE_ Ray get_ray(const Real &u, const Real &v) const {
    return Ray(origin,
               lower_left_corner + u * horizontal + v * vertical - origin);
  }

 private:
  Point3 origin;
  Point3 lower_left_corner;
  Vector3 horizontal;
  Vector3 vertical;
};
}  // namespace rt
}  // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:17:32
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-21 17:19:04
 * @Description:
 */
#pragma once
#include "ray.hpp"

namespace dym {
namespace rt {
template <bool useFocus = true>
class Camera {
 public:
  Camera() {}
  Camera(const Point3& lookfrom, const Point3& lookat, const Vector3& vup,
         const Real& vfov,  // vertical field-of-view in degrees
         const Real& aspect_ratio, const Real& aperture = 2.f,
         Real focus_dist = 1.f, Real f = -1) {
    setCamera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist,
              f);
  }

  _DYM_FORCE_INLINE_ void setCamera(
      const Point3& lookfrom, const Point3& lookat, const Vector3& vup,
      const Real& vfov,  // vertical field-of-view in degrees
      const Real& aspect_ratio, const Real& aperture = 2.f,
      Real focus_dist = 1.f, Real f = -1) {
    auto theta = degrees_to_radians(vfov);
    auto h = tan(theta / 2);
    auto viewport_height = 2.f * h;
    auto viewport_width = aspect_ratio * viewport_height;

    w = (lookfrom - lookat).normalize();
    u = (vup.cross(w)).normalize();
    v = w.cross(u);

    if constexpr (useFocus == false) focus_dist = 1.f;
    origin = lookfrom;
    horizontal = focus_dist * viewport_width * u;
    vertical = focus_dist * viewport_height * v;
    lower_left_corner =
        origin - horizontal * 0.5f - vertical * 0.5f - w * focus_dist;

    lens_radius = aperture / 2;

    Real n =
        -(getViewMatrix4_transform() *
          Vector4((lower_left_corner + 0.5 * (horizontal + vertical)), 1))[2];
    if (f < n) f = 2 * n;

    // Mpersp
    M_persp = Matrix4({
        {1 / (h * aspect_ratio), 0, 0, 0},
        {0, 1 / h, 0, 0},
        {0, 0, f / (f - n), -n * f / (f - n)},
        {0, 0, 1, 0},
    });
  }

  _DYM_FORCE_INLINE_ Ray get_ray(const Real& s, const Real& t) const {
    if constexpr (useFocus) {
      Vector3 rd = lens_radius * random_in_unit_disk();
      Vector3 offset = u * rd.x() + v * rd.y();
      return Ray(origin + offset, lower_left_corner + (1 - s) * horizontal +
                                      (1 - t) * vertical - origin - offset);
    } else
      return Ray(origin, lower_left_corner + (1 - s) * horizontal +
                             (1 - t) * vertical - origin);
  }
  _DYM_FORCE_INLINE_ Matrix4 getViewMatrix4_transform() {
    Vector3& P = origin;
    return Matrix4({Vector4(u, -P.dot(u)), Vector4(v, -P.dot(v)),
                    Vector4(w, -P.dot(w)), Vector4({0, 0, 0, 1})});
  }
  _DYM_FORCE_INLINE_ Matrix4 getViewMatrix4_Perspective() { return M_persp; }

 private:
  Point3 origin;
  Point3 lower_left_corner;
  Vector3 horizontal;
  Vector3 vertical;
  Vector3 u, v, w;
  Real lens_radius;

  Matrix4 M_persp;
};
}  // namespace rt
}  // namespace dym
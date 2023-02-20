/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:17:32
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-21 17:19:04
 * @Description:
 */
#pragma once
#include "dyMath.hpp"
#include "math/define.hpp"
#include "ray.hpp"
#include "tools/sugar.hpp"
#include <tuple>

namespace dym {
_DYM_FORCE_INLINE_ Vector2 st2uv(const Vector2 &st) { return 1 - st; }
_DYM_FORCE_INLINE_ Vector2 uv2st(const Vector2 &uv) { return 1 - uv; }
namespace rt {
class Camera {
public:
  Camera() {}
  Camera(const Point3 &lookfrom, const Point3 &lookat, const Vector3 &vup,
         const Real &vfov, // vertical field-of-view in degrees
         const Real &aspect_ratio, const Real &aperture = 0.,
         Real focus_dist = 1., Real f = -1) {
    setCamera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist,
              f);
  }

  _DYM_FORCE_INLINE_ void
  setCamera(const Point3 &lookfrom, const Point3 &lookat, const Vector3 &vup,
            const Real &vfov, // vertical field-of-view in degrees
            const Real &aspect_ratio, Real aperture = 0., Real focus_dist = 1.,
            Real f = -1.) {
    // if constexpr (useFocus == false)
    //   focus_dist = 1., aperture = .0;
    // save params
    params.lookfrom = lookfrom;
    params.lookat = lookat;
    params.vup = vup;
    params.vfov = vfov;
    params.aspect_ratio = aspect_ratio;
    params.aperture = aperture;
    params.focal_distance = focus_dist;

    auto theta = degrees_to_radians(vfov);
    auto h = tan(theta / 2);
    viewport_height = 2. * focus_dist * h;
    viewport_width = aspect_ratio * viewport_height;
    viewport_area = viewport_height * viewport_width;

    w = (lookfrom - lookat).normalize();
    u = (vup.cross(w)).normalize();
    v = w.cross(u);
    w_inv = -w;

    origin = lookfrom;
    horizontal = focus_dist * viewport_width * u;
    vertical = focus_dist * viewport_height * v;
    lower_left_corner =
        origin - horizontal * 0.5f - vertical * 0.5f - w * focus_dist;

    lens_radius = aperture / 2;

    lens_area = lens_radius > 0 ? (dym::sqr(lens_radius) * dym::Pi) : 1;

    // Mrotate
    Vector3 &P = origin;
    M_transf = Matrix4(
        {Vector4(u, 0), Vector4(v, 0), Vector4(w, 0), Vector4({0, 0, 0, 1})});
    // Mtransf
    M_transf = Matrix4({Vector4(u, -P.dot(u)), Vector4(v, -P.dot(v)),
                        Vector4(w, -P.dot(w)), Vector4({0, 0, 0, 1})});
    // Mpersp
    // Real n =
    //     -(getViewMatrix4_transform() *
    //       Vector4((lower_left_corner + 0.5 * (horizontal + vertical)),
    //       1))[2];
    Real n = focus_dist;
    qprint(n);
    if (f < n)
      f = 2 * n;
    params.f = f;
    // M_persp = Matrix4({
    //     {1 / (focus_dist * h * aspect_ratio), 0, 0, 0},
    //     {0, 1 / (focus_dist * h), 0, 0},
    //     {0, 0, f / (f - n), -n * f / (f - n)},
    //     {0, 0, 1, 0},
    // });
    M_persp = Matrix4({
                  {2, 0, 0, 0},
                  {0, 2, 0, 0},
                  {0, 0, (n + f) / n, 1 / n},
                  {0, 0, -f, 0},
              }) *
              Matrix4({
                  {1 / (h * aspect_ratio), 0, 0, 0},
                  {0, 1 / h, 0, 0},
                  {0, 0, 1, 0},
                  {0, 0, 0, 1},
              });
    qprint(M_persp);
    qprint(Matrix4({
               {1, 0, 0, 0},
               {0, 1, 0, 0},
               {0, 0, (n + f) / n, 1 / n},
               {0, 0, -f, 0},
           }) *
           Matrix4({
               {1 / (h * aspect_ratio), 0, 0, 0},
               {0, 1 / h, 0, 0},
               {0, 0, 1, 0},
               {0, 0, 0, 1},
           }));
  }

  _DYM_FORCE_INLINE_ Ray get_ray(const Real &s, const Real &t) const {
    Vector3 rd = lens_radius * random_in_unit_disk();
    Vector3 offset = u * rd.x() + v * rd.y();
    auto oo = origin + offset;
    return Ray(oo, lower_left_corner + (1 - s) * horizontal +
                       (1 - t) * vertical - oo);
    // if constexpr (useFocus) {
    //   Vector3 rd = lens_radius * random_in_unit_disk();
    //   Vector3 offset = u * rd.x() + v * rd.y();
    //   auto oo = origin + offset;
    //   return Ray(oo, lower_left_corner + (1 - s) * horizontal +
    //                      (1 - t) * vertical - oo);
    // } else
    //   return Ray(origin, lower_left_corner + (1 - s) * horizontal +
    //                          (1 - t) * vertical - origin);
  }

  _DYM_FORCE_INLINE_ auto eval_we(const Ray &r) const {
    auto pos_on_film = r.dir + r.orig;
    auto pos_on_film_rel = pos_on_film - lower_left_corner;
    // auto pofr_len = pos_on_film_rel.length();
    // pos_on_film_rel = pos_on_film_rel.normalize();
    Vector2 coord_uv{
        pos_on_film_rel.dot(u) / (params.focal_distance * viewport_width),
        pos_on_film_rel.dot(v) / (params.focal_distance * viewport_height)};
    // coord_uv *= pofr_len;
    // qprint(pos_on_film, pos_on_film_rel);
    // qprint("hv:", horizontal, vertical);
    // qprint(dym::vector::dot(pos_on_film_rel.cross(horizontal).normalize(),
    //                         pos_on_film_rel.cross(vertical).normalize()));
    // Vector3 pos_on_film =
    //     lower_left_corner + coord[0] * horizontal + coord[1] * vertical;
    // auto offset = pos_on_film - r.dir - origin;
    const Real cos_theta = cos_(r.dir.normalize());
    const Real cos2_theta = dym::sqr(cos_theta);
    const Real we = dym::sqr(params.focal_distance) /
                    (viewport_area * lens_area * dym::sqr(cos2_theta));
    return std::make_tuple(we, coord_uv, w_inv);
  }

  _DYM_FORCE_INLINE_ auto pdf_we(const Ray &r) const {
    const Real cos_theta = cos_(r.dir.normalize());
    const Real cos2_theta = dym::sqr(cos_theta);
    const Real pdf = dym::sqr(params.focal_distance) /
                     (viewport_area * cos_theta * cos2_theta);
    return std::make_tuple(1 / lens_area, pdf);
  }

  _DYM_FORCE_INLINE_ Matrix4 getViewMatrix4_transform() const {
    // Vector3& P = origin;
    // return Matrix4({Vector4(u, -P.dot(u)), Vector4(v, -P.dot(v)),
    //                 Vector4(w, -P.dot(w)), Vector4({0, 0, 0, 1})});
    return M_transf;
  }
  _DYM_FORCE_INLINE_ Matrix4 getViewMatrix4_Perspective() const {
    return M_persp;
  }

  _DYM_FORCE_INLINE_ Matrix4 getViewMatrix4_Rotate() const { return M_rotate; }

private:
  _DYM_FORCE_INLINE_ Real cos_(const Vector3 &d) const { return w_inv.dot(d); }
  struct Params {
    Vector3 lookfrom;
    Vector3 lookat;
    Vector3 vup;
    Real aperture = 1;
    Real vfov = 0; // vertical field-of-view in degrees
    Real aspect_ratio = 0;
    Real focal_distance = 0;
    Real f = -1;
  };
  Params params;
  Point3 origin;
  Point3 lower_left_corner;
  Vector3 horizontal;
  Vector3 vertical;
  Vector3 u, v, w, w_inv;
  Real lens_radius;
  Real viewport_height;
  Real viewport_width;
  Real viewport_area, lens_area;

  Matrix4 M_persp, M_transf, M_rotate;
};
} // namespace rt
} // namespace dym
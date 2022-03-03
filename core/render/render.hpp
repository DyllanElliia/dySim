/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:31:38
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-03 16:12:02
 * @Description:
 */
#pragma once
#include "ray.hpp"
#include "sphere.hpp"
#include "camera.hpp"
#include "hittableList.hpp"

// material
#include "material/lambertian.hpp"
#include "material/metal.hpp"
#include "material/dielectric.hpp"

namespace dym {
namespace rt {
ColorRGB ray_color(const Ray& r, const Hittable& world, int depth) {
  HitRecord rec;

  // If we've exceeded the ray bounce limit, no more light is gathered.
  if (depth <= 0) return ColorRGB(0.f);
  // qprint("111");

  if (world.hit(r, 0.001, infinity, rec)) {
    Ray scattered;
    ColorRGB attenuation;
    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
      return attenuation * ray_color(scattered, world, depth - 1);
    return ColorRGB({0, 0, 0});
  }
  // qprint("222");
  Vector3 unit_direction = r.direction().normalize();
  Real t = 0.5f * (unit_direction.y() + 1.f);
  // qprint(t, (1.f - t) * ColorRGB(1.f) + t * ColorRGB({0.5f, 0.7f, 1.0f}));
  return (1.f - t) * ColorRGB(1.f) + t * ColorRGB({0.5f, 0.7f, 1.0f});
}
}  // namespace rt
}  // namespace dym
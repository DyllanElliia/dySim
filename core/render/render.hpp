/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:31:38
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-07 17:14:22
 * @Description:
 */
#pragma once
#include "ray.hpp"
#include "camera.hpp"
#include "hittableList.hpp"

// Bvh
#include "BVH/bvhNode.hpp"

// object
#include "object/sphere.hpp"

// material
#include "material/lambertian.hpp"
#include "material/metal.hpp"
#include "material/dielectric.hpp"
#include "material/diffuseLight.hpp"

// texture
#include "texture/solidColor.hpp"
#include "texture/imageTexture.hpp"

namespace dym {
namespace rt {
ColorRGB ray_color(const Ray& r, const Hittable& world, int depth) {
  HitRecord rec;
  // If we've exceeded the ray bounce limit, no more light is gathered.
  if (depth <= 0) return ColorRGB(0.f);

  if (world.hit(r, 0.001, infinity, rec)) {
    Ray scattered;
    ColorRGB attenuation;
    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
      return attenuation * ray_color(scattered, world, depth - 1);
    return ColorRGB({0, 0, 0});
  }
  Vector3 unit_direction = r.direction().normalize();
  Real t = 0.5f * (unit_direction.y() + 1.f);
  // qprint(t, (1.f - t) * ColorRGB(1.f) + t * ColorRGB({0.5f, 0.7f, 1.0f}));
  return (1.f - t) * ColorRGB(1.f) + t * ColorRGB({0.5f, 0.7f, 1.0f});
}

ColorRGB ray_color2(
    const Ray& r, const Hittable& world, int depth,
    const std::function<ColorRGB(const Ray& r)>& background = [](const Ray& r) {
      return ColorRGB(0.f);
    }) {
  HitRecord rec;
  // If we've exceeded the ray bounce limit, no more light is gathered.
  if (depth <= 0) return ColorRGB(0.f);

  // If the ray hits nothing, return the background color.
  if (!world.hit(r, 0.001, infinity, rec)) return background(r);

  Ray scattered;
  ColorRGB attenuation;
  ColorRGB emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

  if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered)) return emitted;

  return emitted +
         attenuation * ray_color2(scattered, world, depth - 1, background);
}
}  // namespace rt
}  // namespace dym
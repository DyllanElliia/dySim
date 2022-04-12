/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:31:38
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-12 17:18:11
 * @Description:
 */
#pragma once
#include "camera.hpp"
#include "hittableList.hpp"
#include "ray.hpp"

// Bvh
#include "BVH/bvhNode.hpp"

// object
#include "object/box.hpp"
#include "object/materialObject/constantMedium.hpp"
#include "object/sphere.hpp"
#include "object/transform.hpp"
#include "object/triangle.hpp"
#include "object/mesh.hpp"

// material
#include "material/dielectric.hpp"
#include "material/diffuseLight.hpp"
#include "material/lambertian.hpp"
#include "material/metal.hpp"

// texture
#include "texture/imageTexture.hpp"
#include "texture/solidColor.hpp"

// pdf
#include "pdf/pdf.hpp"

namespace dym {
namespace rt {
// ColorRGB ray_color(const Ray& r, const Hittable& world, int depth) {
//   HitRecord rec;
//   // If we've exceeded the ray bounce limit, no more light is gathered.
//   if (depth <= 0) return ColorRGB(0.f);

//   if (world.hit(r, 0.001, infinity, rec)) {
//     Ray scattered;
//     ColorRGB attenuation;

//     Real asdf;
//     if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, asdf))
//       return attenuation * ray_color(scattered, world, depth - 1);
//     return ColorRGB({0, 0, 0});
//   }
//   Vector3 unit_direction = r.direction().normalize();
//   Real t = 0.5f * (unit_direction.y() + 1.f);
//   // qprint(t, (1.f - t) * ColorRGB(1.f) + t * ColorRGB({0.5f, 0.7f, 1.0f}));
//   return (1.f - t) * ColorRGB(1.f) + t * ColorRGB({0.5f, 0.7f, 1.0f});
// }

// ColorRGB ray_color2(
//     const Ray& r, const Hittable& world, int depth,
//     const std::function<ColorRGB(const Ray& r)>& background = [](const Ray&
//     r) {
//       return ColorRGB(0.f);
//     }) {
//   HitRecord rec;
//   // If we've exceeded the ray bounce limit, no more light is gathered.
//   if (depth <= 0) return ColorRGB(0.f);

//   // If the ray hits nothing, return the background color.
//   if (!world.hit(r, 0.001, infinity, rec)) return background(r);

//   Ray scattered;
//   ColorRGB attenuation;
//   ColorRGB emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

//   Real asdf;
//   if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered, asdf))
//     return emitted;

//   return emitted +
//          attenuation * ray_color2(scattered, world, depth - 1, background);
// }

ColorRGB ray_color_pdf(
    const Ray& r, const Hittable& world, shared_ptr<HittableList> lights,
    int depth,
    const std::function<ColorRGB(const Ray& r)>& background = [](const Ray& r) {
      return ColorRGB(0.f);
    }) {
  HitRecord rec;
  // If we've exceeded the ray bounce limit, no more light is gathered.
  if (depth <= 0) return ColorRGB(0.f);

  // If the ray hits nothing, return the background color.
  if (!world.hit(r, 0.001, infinity, rec)) return background(r);

  ScatterRecord srec;
  ColorRGB emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
  if (!rec.mat_ptr->scatter(r, rec, srec)) return emitted;

  if (srec.is_specular) {
    return srec.attenuation * ray_color_pdf(srec.specular_ray, world, lights,
                                            depth - 1, background);
  }

  shared_ptr<pdf> p;
  if (lights) {
    auto light_ptr = make_shared<hittable_pdf>(lights, rec.p);
    p = make_shared<mixture_pdf>(light_ptr, srec.pdf_ptr);
  } else
    p = srec.pdf_ptr;

  Ray scattered = Ray(rec.p, p->generate(), r.time());
  auto pdf_val = p->value(scattered.direction());

  return emitted +
         srec.attenuation * rec.mat_ptr->scattering_pdf(r, rec, scattered) *
             ray_color_pdf(scattered, world, lights, depth - 1, background) /
             pdf_val;
}
}  // namespace rt
}  // namespace dym
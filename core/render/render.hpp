/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:31:38
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-18 18:22:33
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
struct GBuffer {
  Vector3 normal;
  Vector3 positionn;
  Vector3 albedo;
  int obj_id;
  GBuffer(int asdf = 0) {}
};

namespace {
GBuffer globalGBuffer;  // only for template, don't use it!
}

template <bool recordHitRecord = false>
ColorRGB ray_color_pdf(
    const Ray& r, const Hittable& world, shared_ptr<HittableList> lights,
    int depth,
    const std::function<ColorRGB(const Ray& r)>& background =
        [](const Ray& r) { return ColorRGB(0.f); },
    GBuffer& out_gbuffer = globalGBuffer) {
  HitRecord rec;
  // If we've exceeded the ray bounce limit, no more light is gathered.
  if (depth <= 0) return ColorRGB(0.f);

  // If the ray hits nothing, return the background color.
  if (!world.hit(r, 0.001, infinity, rec)) return background(r);

  ScatterRecord srec;
  ColorRGB emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

  if constexpr (recordHitRecord) {
    out_gbuffer.normal = rec.normal;
    out_gbuffer.position = rec.position;
    out_gbuffer.albedo = emitted;
    out_gbuffer.obj_id = rec.obj_id;
  }

  if (!rec.mat_ptr->scatter(r, rec, srec)) return emitted;

  if constexpr (recordHitRecord) out_gbuffer.albedo += srec.attenuation;

  if (srec.is_specular) {
    return srec.attenuation * ray_color_pdf(srec.specular_ray, world, lights,
                                            depth - 1, background,
                                            globalGBuffer);
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

template <bool cameraUseFocus = false>
class RtRender {
 public:
  RtRender(const int& image_width, const int& image_height)
      : image_width(image_width),
        image_height(image_height),
        aspect_ratio(image_width / Real(image_height)),
        image(Tensor<dym::Vector<Real, dym::PIC_RGB>>(
            0, dym::gi(image_height, image_width))),
        image_GBuffer(
            Tensor<GBuffer, false>(0, dym::gi(image_height, image_width))) {}
  void render(
      int samples_per_pixel, int max_depth,
      const std::function<ColorRGB(const Ray& r)>& background =
          [](const Ray& r) { return ColorRGB(0.f); }) {
    image.for_each_i([&](dym::Vector<Real, dym::PIC_RGB>& color, int i, int j) {
      auto color_pre = color;
      GBuffer gbuffer;
      color = 0.f;
      auto u = (Real)j / (image_width - 1);
      auto v = (Real)i / (image_height - 1);
      dym::rt::Ray r = cam.get_ray(u, v);
      for (int samples = 0; samples < samples_per_pixel; samples++) {
        color += ray_color_pdf<true>(r, *worlds, lights, max_depth, background,
                                     gbuffer);
      }
      image_GBuffer[image.getIndexInt(gi(i, j))] = gbuffer;
      color = color * (1.f / Real(samples_per_pixel));
      color = dym::clamp(dym::sqrt(color) * 255.f, 0.0, 255.99);
      dym::Loop<int, 3>([&](auto pi) {
        if (dym::isnan(color[pi])) color[pi] = 0;
        if (dym::isinf(color[pi])) color[pi] = color_pre[pi];
      });
    });
  }

 private:
  const Real aspect_ratio;
  const int image_width;
  const int image_height;
  Tensor<dym::Vector<Real, dym::PIC_RGB>> image;
  Tensor<GBuffer, false> image_GBuffer;

 public:
  shared_ptr<HittableList> worlds;
  shared_ptr<HittableList> lights;
  Camera<false> cam;
};

}  // namespace rt
}  // namespace dym
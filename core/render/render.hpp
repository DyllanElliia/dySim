/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:31:38
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-05-23 15:26:40
 * @Description:
 */
#pragma once
#include "camera.hpp"
#include "dyMath.hpp"
#include "hittableList.hpp"
#include "randomFun.hpp"
#include "ray.hpp"

// Bvh
#include "BVH/bvhNode.hpp"

// object
#include "object/box.hpp"
#include "object/materialObject/constantMedium.hpp"
#include "object/mesh.hpp"
#include "object/skybox.hpp"
#include "object/sphere.hpp"
#include "object/transform.hpp"
#include "object/triangle.hpp"

// material
#include "material/dielectric.hpp"
#include "material/diffuseLight.hpp"
#include "material/disneryBRDF.hpp"
#include "material/lambertian.hpp"
#include "material/metal.hpp"

// texture
#include "texture/imageTexture.hpp"
#include "texture/solidColor.hpp"

// pdf
#include "pdf/pdf.hpp"

// denoise
#include "denoise/svgf.hpp"

// RenderKernel
#include "renderKernel/MIS_RR_PT.hpp"
#include "renderKernel/rKernel.hpp"

namespace dym {
namespace rt {

namespace {
GBuffer globalGBuffer; // only for template, don't use it!
}

template <bool recordHitRecord = false>
ColorRGB ray_color_pdf(
    const Ray &r, const Hittable &world, shared_ptr<HittableList> lights,
    int depth,
    const std::function<ColorRGB(const Ray &r)> &background =
        [](const Ray &r) { return ColorRGB(0.f); },
    GBuffer &out_gbuffer = globalGBuffer) {
  HitRecord rec;
  // If we've exceeded the ray bounce limit, no more light is gathered.
  if (depth <= 0)
    return ColorRGB(0.f);

  // If the ray hits nothing, return the background color.
  if (!world.hit(r, 0.001, infinity, rec))
    return background(r);

  ScatterRecord srec;
  ColorRGB emitted = rec.mat_ptr->emitted(r, rec);

  if constexpr (recordHitRecord) {
    out_gbuffer.normal = rec.normal;
    out_gbuffer.position = rec.p;
    out_gbuffer.albedo = emitted;
    out_gbuffer.obj_id = rec.obj_id;
  }

  if (!rec.mat_ptr->scatter(r, rec, srec))
    return emitted;

  if constexpr (recordHitRecord)
    out_gbuffer.albedo += srec.attenuation;

  if (srec.is_specular) {
    return srec.attenuation * ray_color_pdf(srec.specular_ray, world, lights,
                                            depth - 1, background,
                                            globalGBuffer);
  }

  shared_ptr<pdf> p;
  if (lights && lights->objects.size() > 0) {
    auto light_ptr = make_shared<hittable_pdf>(lights, rec.p);
    p = make_shared<mixture_pdf>(light_ptr, srec.pdf_ptr, 0.5);
  } else
    p = srec.pdf_ptr;

  Ray scattered = Ray(rec.p, p->generate(), r.time());
  auto pdf_val = p->value(scattered.direction());

  return emitted +
         rec.mat_ptr->BxDF_Evaluate(r, scattered, rec, srec) *
             ray_color_pdf(scattered, world, lights, depth - 1, background) /
             pdf_val;
}

template <bool cameraUseFocus = false> class RtRender {
public:
  RtRender(const int &image_width, const int &image_height)
      : image_width(image_width), image_height(image_height),
        aspect_ratio(image_width / Real(image_height)),
        image(Tensor<dym::Vector<Real, dym::PIC_RGB>>(
            0, dym::gi(image_width, image_height))),
        imageP(Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>>(
            0, dym::gi(image_width, image_height))),
        image_GBuffer(
            Tensor<GBuffer, false>(0, dym::gi(image_width, image_height))) {
    init_svgf(image.shape());
  }

  template <class RenderKernel = RKernel>
  void render(
      int samples_per_pixel, Real endValue,
      const std::function<ColorRGB(const Ray &r)> &background =
          [](const Ray &r) { return ColorRGB(0.f); },
      const Real &max_color = 1.0, dym::Vector3i patchSize = {15, 15}) {
    // auto viewMatrix = cam.getViewMatrix4_transform();
    // Matrix3 viewMatrix3 = viewMatrix;
    // viewMatrix = cam.getViewMatrix4_Perspective() * viewMatrix;
    RenderKernel rk;

    image.for_each_p(
        [&](dym::Vector<Real, dym::PIC_RGB> &color, int i, int j) {
          auto color_pre = color;
          GBuffer gbuffer;
          color = 0.f;
          auto u = (Real)i / (image_width - 1);
          auto v = (Real)j / (image_height - 1);
          dym::rt::Ray r = cam.get_ray(u, v);
          gbuffer = rk.renderGBuffer(r, worlds);
          for (int samples = 0; samples < samples_per_pixel; samples++) {
            color_pre = color;
            color += rk.render(r, worlds,
                               std::make_shared<dym::rt::HittableList>(lights),
                               endValue, background);
            dym::Loop<int, 3>([&](auto pi) {
              if (dym::isnan(color[pi]))
                color[pi] = color_pre[pi] * (samples + 1) / samples;
            });
          }
          // auto pos4 = viewMatrix * Vector4(gbuffer.position, 1);
          // gbuffer.position = pos4 / pos4[3];
          // gbuffer.normal = viewMatrix3 * gbuffer.normal;
          image_GBuffer[image.getIndexInt(gi(i, j))] = gbuffer;
          color = color * (1.f / Real(samples_per_pixel));
          color = dym::sqrt(color) * max_color;
          dym::Loop<int, 3>([&](auto pi) {
            if (dym::isnan(color[pi]))
              color[pi] = 0;
            if (dym::isinf(color[pi]))
              color[pi] = color_pre[pi];
            if (color[pi] > max_color)
              color[pi] = max_color;
          });
        },
        patchSize);
  }

  Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> &
  getFrame(Real PixelScale = 255.) {
    imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB> &e, int i) {
      auto color = dym::clamp(image[i] * PixelScale, 0.0, 255.);
      e = color.cast<dym::Pixel>();
    });
    return imageP;
  }

  Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> &
  getFrameGBuffer(std::string GBuffer_type, Real posScale = 255) {
    auto viewMatrix = cam.getViewMatrix4_transform();
    Matrix3 viewMatrix3 = viewMatrix;
    viewMatrix = cam.getViewMatrix4_Perspective() * viewMatrix;
    switch (hash_(GBuffer_type.c_str())) {
    case hash_compile_time("normal"):
      imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB> &e, int i) {
        auto pix = (viewMatrix3 * image_GBuffer[i].normal + 1) / 2.0;
        pix = dym::clamp(pix * 255, 0.0, 255.99);
        e = pix.cast<dym::Pixel>();
        e[2] = 255;
      });
      break;
    case hash_compile_time("position"):
      imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB> &e, int i) {
        auto pix = viewMatrix * Vector4(image_GBuffer[i].position, 1.0);
        pix = dym::clamp((1.0 + pix / pix[3]) / 2 * posScale, 0.0, 255.99);
        e = Vector3(pix).cast<dym::Pixel>();
        e[2] = 255;
      });
      break;
    case hash_compile_time("depth"):
      imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB> &e, int i) {
        auto pix = viewMatrix * Vector4(image_GBuffer[i].position, 1.0);
        if (random_real() < 1e-5)
          qprint(pix);

        e = (dym::Pixel)dym::clamp(dym::abs(pix[2] / pix[3]) * posScale, 0.0,
                                   255.99);
      });
      break;
    case hash_compile_time("albedo"):
      imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB> &e, int i) {
        auto pix = image_GBuffer[i].albedo;
        pix = dym::clamp(pix * 255, 0.0, 255.99);
        e = pix.cast<dym::Pixel>();
      });
      break;
    case hash_compile_time("objId"):
      imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB> &e, int i) {
        e = (dym::Pixel)image_GBuffer[i].obj_id;
      });
      break;
    default:
      DYM_ERROR((std::string(
                     "DYM::RT::RtRender.getFrameGbuffer ERROR: failed to find "
                     "GBuffer_type(\"") +
                 GBuffer_type + "\")")
                    .c_str());
    }
    return imageP;
  }

  void denoise() {
    denoise_svgf(image, image_GBuffer);
    svgf_viewMatrix =
        cam.getViewMatrix4_Perspective() * cam.getViewMatrix4_transform();
  }

private:
  const Real aspect_ratio;
  const int image_width;
  const int image_height;
  Tensor<dym::Vector<Real, dym::PIC_RGB>> image;
  Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> imageP;
  Tensor<GBuffer, false> image_GBuffer;

public:
  HittableList worlds;
  HittableList lights;
  Camera<cameraUseFocus> cam;
};

} // namespace rt
} // namespace dym
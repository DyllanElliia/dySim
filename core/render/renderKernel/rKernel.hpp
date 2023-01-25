#pragma once

#include "../render.hpp"
#include "dyMath.hpp"
#include <functional>

namespace dym {
namespace rt {
template <bool cameraUseFocus = false> struct RtMessage {

  std::string name;
  const Real aspect_ratio;
  const int image_width;
  const int image_height;
  Tensor<dym::Vector<Real, dym::PIC_RGB>> &image;
  Tensor<GBuffer, false> &image_GBuffer;
  HittableList &worlds;
  HittableList &lights;
  Camera<cameraUseFocus> &cam;
  RtMessage(std::string name, const Real aspect_ratio, const int image_width,
            const int image_height,
            Tensor<dym::Vector<Real, dym::PIC_RGB>> &image,
            Tensor<GBuffer, false> &image_GBuffer, HittableList &worlds,
            HittableList &lights, Camera<cameraUseFocus> &cam)
      : name(name), aspect_ratio(aspect_ratio), image_width(image_width),
        image_height(image_height), image(image), image_GBuffer(image_GBuffer),
        worlds(worlds), lights(lights), cam(cam) {}
};

template <bool cameraUseFocus = false> class RKernel {
public:
  RKernel(){};
  ~RKernel(){};
  virtual GBuffer renderGBuffer(const Ray &r, const Hittable &world) {
    GBuffer out_gbuffer;
    HitRecord rec;
    if (!world.hit(r, 0.001, infinity, rec))
      return out_gbuffer;
    ScatterRecord srec;
    ColorRGB emitted = rec.mat_ptr->emitted(r, rec);
    out_gbuffer.normal = rec.normal;
    out_gbuffer.position = rec.p;
    out_gbuffer.albedo = emitted;
    out_gbuffer.obj_id = rec.obj_id;
    if (rec.mat_ptr->scatter(r, rec, srec))
      out_gbuffer.albedo += srec.attenuation;
    out_gbuffer.albedo = min(out_gbuffer.albedo, Vector3(1.));
    return out_gbuffer;
  }

  virtual bool endCondition(Real &value) { return value <= 0 ? true : false; }

  virtual ColorRGB render(
      const Ray &r, const Hittable &world, shared_ptr<HittableList> lights,
      Real depth,
      const std::function<ColorRGB(const Ray &r)> &background =
          [](const Ray &r) { return ColorRGB(0.f); }) {
    HitRecord rec;
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (endCondition(depth))
      return ColorRGB(0.f);

    // If the ray hits nothing, return the background color.
    if (!world.hit(r, 0.001, infinity, rec))
      return background(r);

    ScatterRecord srec;
    ColorRGB Le = rec.mat_ptr->emitted(r, rec);

    if (!rec.mat_ptr->scatter(r, rec, srec))
      return Le;

    if (srec.is_specular && !srec.pdf_ptr) {
      Ray scattered_ = srec.specular_ray;
      return Le + rec.mat_ptr->BxDF_Evaluate(r, scattered_, rec, srec) *
                      render(scattered_, world, lights, depth - 1, background);
    }

    shared_ptr<pdf> matPdf;
    if (lights && lights->objects.size() > 0) {
      auto light_ptr = make_shared<hittable_pdf>(lights, rec.p);
      matPdf =
          make_shared<mixture_pdf>(light_ptr, srec.pdf_ptr, srec.is_specular);
    } else
      matPdf = srec.pdf_ptr;

    Ray scattered = Ray(rec.p, matPdf->generate(), r.time());
    auto Fr = rec.mat_ptr->BxDF_Evaluate(r, scattered, rec, srec);
    auto pdf_val = matPdf->value(scattered.direction());
    ColorRGB Li = render(scattered, world, lights, depth - 1, background);
    return Le + Fr * Li / pdf_val;
  }
  virtual void test() { qprint("test"); }
  virtual void
  runKernel(RtMessage<cameraUseFocus> &rm, int samples_per_pixel, Real endValue,
            const std::function<ColorRGB(const Ray &r)> &background,
            const Real &max_color, dym::Vector3i patchSize) {
    rm.image.for_each_p(
        [&](dym::Vector<Real, dym::PIC_RGB> &color, int i, int j) {
          auto color_pre = color;
          GBuffer gbuffer;
          color = 0.f;
          auto u = (Real)i / (rm.image_width - 1);
          auto v = (Real)j / (rm.image_height - 1);
          dym::rt::Ray r = rm.cam.get_ray(u, v);
          gbuffer = renderGBuffer(r, rm.worlds);
          for (int samples = 0; samples < samples_per_pixel; samples++) {
            color_pre = color;
            color += render(r, rm.worlds,
                            std::make_shared<dym::rt::HittableList>(rm.lights),
                            endValue, background);
            dym::Loop<int, 3>([&](auto pi) {
              if (dym::isnan(color[pi]))
                color[pi] = color_pre[pi] * (samples + 1) / samples;
            });
          }
          // auto pos4 = viewMatrix * Vector4(gbuffer.position, 1);
          // gbuffer.position = pos4 / pos4[3];
          // gbuffer.normal = viewMatrix3 * gbuffer.normal;
          rm.image_GBuffer[rm.image.getIndexInt(gi(i, j))] = gbuffer;
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
};

} // namespace rt
} // namespace dym
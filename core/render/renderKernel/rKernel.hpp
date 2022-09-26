#pragma once

#include "../render.hpp"
#include "dyMath.hpp"
#include <functional>

namespace dym {
namespace rt {

class RKernel {
public:
  RKernel(){};
  ~RKernel(){};
  virtual GBuffer renderGBuffer(const Ray &r, const Hittable &world) {
    GBuffer out_gbuffer;
    HitRecord rec;
    if (!world.hit(r, 0.001, infinity, rec))
      return out_gbuffer;
    ScatterRecord srec;
    ColorRGB emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
    out_gbuffer.normal = rec.normal;
    out_gbuffer.position = rec.p;
    out_gbuffer.albedo = emitted;
    out_gbuffer.obj_id = rec.obj_id;
    if (rec.mat_ptr->scatter(r, rec, srec))
      out_gbuffer.albedo += srec.attenuation;
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
    ColorRGB Le = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

    if (!rec.mat_ptr->scatter(r, rec, srec))
      return Le;

    if (srec.is_light)
      return render(srec.specular_ray, world, lights, depth - 1, background);

    if (srec.is_specular && !srec.pdf_ptr) {
      Ray scattered_ = srec.specular_ray;
      return Le + rec.mat_ptr->BRDF_Evaluate(r, scattered_, rec, srec) *
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
    auto Fr = rec.mat_ptr->BRDF_Evaluate(r, scattered, rec, srec);
    auto pdf_val = matPdf->value(scattered.direction());
    ColorRGB Li = render(scattered, world, lights, depth - 1, background);
    return Le + Fr * Li / pdf_val;
  }
};

} // namespace rt
} // namespace dym
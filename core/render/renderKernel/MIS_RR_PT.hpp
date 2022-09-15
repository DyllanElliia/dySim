#pragma once

#include "rKernel.hpp"
#include "render/randomFun.hpp"
#include <cstdlib>

namespace dym {
namespace rt {
class MIS_RR_PT : public RKernel {
  virtual bool endCondition(Real &value) { return random_real() > value; }

  virtual ColorRGB render(
      const Ray &r, const Hittable &world, shared_ptr<HittableList> lights,
      Real RR,
      const std::function<ColorRGB(const Ray &r)> &background =
          [](const Ray &r) { return ColorRGB(0.f); }) {
    if (RR > 1)
      RR = 0.1;
    HitRecord rec;
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (endCondition(RR))
      return ColorRGB(0.f);

    // If the ray hits nothing, return the background color.
    if (!world.hit(r, 0.001, infinity, rec))
      return background(r) / RR;

    ScatterRecord srec;
    ColorRGB emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

    if (!rec.mat_ptr->scatter(r, rec, srec))
      return emitted / RR;

    if (srec.is_specular) {
      return (emitted + srec.attenuation * render(srec.specular_ray, world,
                                                  lights, RR, background)) /
             RR;
    }

    shared_ptr<pdf> p;
    if (lights && lights->objects.size() > 0) {
      auto light_ptr = make_shared<hittable_pdf>(lights, rec.p);
      p = make_shared<mixture_pdf>(light_ptr, srec.pdf_ptr);
    } else
      p = srec.pdf_ptr;

    Ray scattered = Ray(rec.p, p->generate(), r.time());
    auto pdf_val = p->value(scattered.direction());

    auto Li = render(scattered, world, lights, RR, background);
    return (emitted + srec.attenuation *
                          rec.mat_ptr->BRDF_Evaluate(r, scattered, rec, srec) *
                          Li / pdf_val) /
           RR;
  }
};
} // namespace rt
} // namespace dym
#pragma once

#include "rKernel.hpp"
#include "render/randomFun.hpp"
#include <cstdlib>

namespace dym {
namespace rt {
class MIS_RR_PT : public RKernel {
public:
  virtual bool endCondition(Real &value) { return random_real() > value; }

  virtual ColorRGB render(
      const Ray &r, const Hittable &world, shared_ptr<HittableList> lights,
      Real RR,
      const std::function<ColorRGB(const Ray &r)> &background =
          [](const Ray &r) { return ColorRGB(0.f); }) {
    if (RR > 1)
      RR = 0.9;
    HitRecord rec;
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (endCondition(RR))
      return ColorRGB(0.f);

    // If the ray hits nothing, return the background color.
    if (!world.hit(r, 0.001, infinity, rec))
      return background(r) / RR;

    ScatterRecord srec;
    ColorRGB Le = rec.mat_ptr->emitted(r, rec);

    if (!rec.mat_ptr->scatter(r, rec, srec))
      return Le / RR;

    if (srec.is_specular && !srec.pdf_ptr) {
      Ray scattered_ = srec.specular_ray;
      return (Le + rec.mat_ptr->BRDF_Evaluate(r, scattered_, rec, srec) *
                       render(scattered_, world, lights, RR, background)) /
             RR;
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
    ColorRGB Li = render(scattered, world, lights, RR, background);
    return (Le + Fr * Li / pdf_val) / RR;
  }
};
} // namespace rt
} // namespace dym
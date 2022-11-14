/*
 * @Author: DyllanElliia
 * @Date: 2022-03-03 15:28:54
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-23 16:51:58
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
#include "render/pdf/pdf.hpp"
#include "render/randomFun.hpp"
#include <cstddef>
namespace dym {
namespace rt {
class Metal : public Material {
public:
  Metal(const ColorRGB &color, const Real &fuzz = -1.f)
      : albedo(make_shared<SolidColor>(color)), fuzz(fuzz <= 1.f ? fuzz : 1.f) {
  }
  Metal(const shared_ptr<Texture> &tex, const Real &fuzz = 0.)
      : albedo(tex), fuzz(fuzz) {}

  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const override {
    Vector3 reflected = r_in.direction().normalize().reflect(rec.normal);
    srec.specular_ray = Ray(rec.p, reflected);
    srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
    srec.is_specular = fuzz > 0.01 ? false : true;
    srec.pdf_ptr =
        fuzz != 0 ? make_shared<GTR2_pdf>(rec.normal, fuzz, r_in.direction())
                  : nullptr;

    return true;
  }

  virtual Vector3 BxDF_Evaluate(const Ray &r_in, const Ray &scattered,
                                const HitRecord &rec,
                                const ScatterRecord &srec) const {
    Vector3 L = scattered.direction(), V = -r_in.direction();
    Vector3 N = rec.normal;
    Vector3 H = (L + V).normalize();
    auto NdotH = N.dot(H);
    auto LdotH = L.dot(H);
    auto NdotL = N.dot(L);
    auto NdotV = N.dot(V);
    if (NdotL < 0 || NdotV < 0)
      return 0.0;
    auto alphaG = sqr(0.5 + fuzz / 2.0);
    auto Ds = solve_GTR2_pdf(NdotH, alphaG);
    auto FH = SchlickFresnel(LdotH);
    auto Gs = smithG_GGX(NdotV, alphaG) * smithG_GGX(NdotL, alphaG);

    ColorRGB Fs = lerp(srec.attenuation, ColorRGB(1.0), clamp(FH, 0.0, 1.0));
    if (fuzz < 0.01)
      return Fs;

    return Gs * Ds * Fs;
    // return Gs * Ds;
    // return 1;
  }

public:
  shared_ptr<Texture> albedo;
  Real fuzz;
};
} // namespace rt
} // namespace dym
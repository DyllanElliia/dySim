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
      : albedo(tex), fuzz(fuzz <= 1. && fuzz >= 0.01 ? fuzz : 0.) {}

  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const override {
    // Vector3 nor = rec.normal;
    // if (fuzz > 0) nor += random_unit_vector() * fuzz;

    // Vector3 reflected = r_in.direction().normalize().reflect(nor);
    // scattered = Ray(rec.p, reflected);
    // attenuation = albedo->value(rec.u, rec.v, rec.p);

    // pdf = 1;

    // return (scattered.direction().dot(rec.normal) > 0);

    Vector3 reflected = r_in.direction().normalize().reflect(rec.normal);
    srec.specular_ray = Ray(rec.p, reflected);
    srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
    srec.is_specular = true;
    srec.pdf_ptr =
        fuzz != 0 ? make_shared<GTR2_pdf>(rec.normal, fuzz, r_in.direction())
                  : nullptr;

    return true;
  }

  virtual Real scattering_pdf(const Ray &r_in, const HitRecord &rec,
                              const Ray &scattered) const {
    if (fuzz == 0)
      return 1.;
    Vector3 L = scattered.direction(), V = -r_in.direction();
    Vector3 N = rec.normal;
    Vector3 H = (scattered.direction() - r_in.direction()).normalize();
    auto NdotH = N.dot(H);
    auto LdotH = L.dot(H);
    auto NdotL = N.dot(L);
    auto NdotV = N.dot(V);
    auto Ds = solve_GTR2_pdf(NdotH, sqr(fuzz));
    auto Fs = SchlickFresnel(LdotH);
    auto Gs = smithG_GGX(NdotV, fuzz) * smithG_GGX(NdotL, fuzz);
    // return Gs * Fs * Ds;
    return Gs * Ds;
    // return 1;
  }

private:
  _DYM_FORCE_INLINE_ Real SchlickFresnel(const Real &u) const {
    Real m = clamp(1 - u, 0, 1);
    Real m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
  }

  _DYM_FORCE_INLINE_ Real smithG_GGX(const Real &NdotV,
                                     const Real &alphaG) const {
    Real a = alphaG * alphaG;
    Real b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
  }

public:
  shared_ptr<Texture> albedo;
  Real fuzz;
};
} // namespace rt
} // namespace dym
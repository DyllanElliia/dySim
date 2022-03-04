/*
 * @Author: DyllanElliia
 * @Date: 2022-03-03 15:56:56
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-04 15:55:14
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
namespace dym {
namespace rt {
namespace {
_DYM_FORCE_INLINE_ Vector3 refract(const Vector3& uv, const Vector3& n,
                                   const Real& etai_over_etat) {
  auto cos_theta = fmin(dym::vector::dot(-uv, n), 1.f);
  Vector3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
  Vector3 r_out_parallel =
      Real(-dym::sqrt(dym::abs(1.0 - r_out_perp.length_sqr()))) * n;
  return r_out_perp + r_out_parallel;
}
}  // namespace

class Dielectric : public Material {
 public:
  Dielectric(const Real& index_of_refraction)
      : ir(index_of_refraction),
        albedo(make_shared<SolidColor>(ColorRGB(1.f))) {}
  Dielectric(const ColorRGB& color, const Real& index_of_refraction)
      : ir(index_of_refraction), albedo(make_shared<SolidColor>(color)) {}
  Dielectric(const shared_ptr<Texture>& tex, const Real& index_of_refraction)
      : ir(index_of_refraction), albedo(tex) {}

  virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                       ColorRGB& attenuation, Ray& scattered) const override {
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    Real refraction_ratio = rec.front_face ? (1.f / ir) : ir;

    Vector3 unit_direction = r_in.direction().normalize();
    Real cos_theta = fmin(dym::vector::dot(-unit_direction, rec.normal), 1.f);
    Real sin_theta = dym::sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.f;
    Vector3 direction;

    if (cannot_refract ||
        reflectance(cos_theta, refraction_ratio) > random_real())
      direction = unit_direction.reflect(rec.normal);
    else
      direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = Ray(rec.p, direction);

    return true;
  }

 public:
  Real ir;  // Index of Refraction
  shared_ptr<Texture> albedo;

 private:
  static _DYM_FORCE_INLINE_ Real reflectance(const Real& cosine,
                                             const Real& ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
  }
};
}  // namespace rt
}  // namespace dym
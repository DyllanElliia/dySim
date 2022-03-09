/*
 * @Author: DyllanElliia
 * @Date: 2022-03-03 15:28:54
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-09 17:05:59
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
namespace dym {
namespace rt {
class Metal : public Material {
 public:
  Metal(const ColorRGB& color, const Real& fuzz = -1.f)
      : albedo(make_shared<SolidColor>(color)),
        fuzz(fuzz <= 1.f ? fuzz : 1.f) {}
  Metal(const shared_ptr<Texture>& tex, const Real& fuzz = -1.f)
      : albedo(tex), fuzz(fuzz <= 1.f ? fuzz : 1.f) {}

  virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                       ColorRGB& attenuation, Ray& scattered) const override {
    Vector3 nor = rec.normal;
    if (fuzz > 0) nor += random_unit_vector() * fuzz;

    Vector3 reflected = r_in.direction().normalize().reflect(nor);
    scattered = Ray(rec.p, reflected);
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return (scattered.direction().dot(rec.normal) > 0);
  }

 public:
  shared_ptr<Texture> albedo;
  Real fuzz;
};
}  // namespace rt
}  // namespace dym
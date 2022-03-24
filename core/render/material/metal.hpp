/*
 * @Author: DyllanElliia
 * @Date: 2022-03-03 15:28:54
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-23 16:51:58
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
                       ScatterRecord& srec) const override {
    // Vector3 nor = rec.normal;
    // if (fuzz > 0) nor += random_unit_vector() * fuzz;

    // Vector3 reflected = r_in.direction().normalize().reflect(nor);
    // scattered = Ray(rec.p, reflected);
    // attenuation = albedo->value(rec.u, rec.v, rec.p);

    // pdf = 1;

    // return (scattered.direction().dot(rec.normal) > 0);

    Vector3 reflected = r_in.direction().normalize().reflect(rec.normal);
    srec.specular_ray =
        Ray(rec.p,
            fuzz > 0 ? reflected + fuzz * random_in_unit_sphere() : reflected);
    srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
    srec.is_specular = true;
    srec.pdf_ptr = nullptr;
    return true;
  }

 public:
  shared_ptr<Texture> albedo;
  Real fuzz;
};
}  // namespace rt
}  // namespace dym
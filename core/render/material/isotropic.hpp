/*
 * @Author: DyllanElliia
 * @Date: 2022-03-15 15:44:07
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-24 14:55:49
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
namespace dym {
namespace rt {
class isotropic : public Material {
public:
  isotropic(ColorRGB c) : albedo(make_shared<SolidColor>(c)) {}
  isotropic(shared_ptr<Texture> a) : albedo(a) {}

  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const override {
    // scattered = Ray(rec.p, random_in_unit_sphere(), r_in.time());
    // attenuation = albedo->value(rec.u, rec.v, rec.p);

    // pdf = 1;

    // return true;

    srec.is_specular = true;
    srec.pdf_ptr = nullptr;
    srec.specular_ray = Ray(rec.p, random_in_unit_sphere(), r_in.time());
    srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
  }

  virtual Vector3 BRDF_Evaluate(const Ray &r_in, const Ray &scattered,
                                const HitRecord &rec,
                                const ScatterRecord &srec) const {
    return srec.attenuation;
  }

public:
  shared_ptr<Texture> albedo;
};
} // namespace rt
} // namespace dym
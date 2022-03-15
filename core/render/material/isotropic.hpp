/*
 * @Author: DyllanElliia
 * @Date: 2022-03-15 15:44:07
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-15 15:48:09
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

  virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                       ColorRGB& attenuation, Ray& scattered) const override {
    scattered = Ray(rec.p, random_in_unit_sphere(), r_in.time());
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
  }

 public:
  shared_ptr<Texture> albedo;
};
}  // namespace rt
}  // namespace dym
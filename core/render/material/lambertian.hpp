/*
 * @Author: DyllanElliia
 * @Date: 2022-03-03 15:24:14
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-10 17:23:35
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
namespace dym {
namespace rt {
class Lambertian : public Material {
 public:
  Lambertian(const ColorRGB& a) : albedo(make_shared<SolidColor>(a)) {}
  Lambertian(const shared_ptr<Texture>& a) : albedo(a) {}

  virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                       ColorRGB& attenuation, Ray& scattered) const override {
    auto scatter_direction = rec.normal + random_unit_vector();

    // Catch degenerate scatter direction
    if (scatter_direction == 0.0) scatter_direction = rec.normal;

    scattered = Ray(rec.p, scatter_direction);
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
  }

 public:
  shared_ptr<Texture> albedo;
};
}  // namespace rt
}  // namespace dym
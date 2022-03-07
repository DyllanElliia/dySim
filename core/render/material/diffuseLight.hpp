/*
 * @Author: DyllanElliia
 * @Date: 2022-03-07 15:27:21
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-07 15:29:43
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
namespace dym {
namespace rt {
class DiffuseLight : public Material {
 public:
  DiffuseLight(shared_ptr<Texture> a) : emit(a) {}
  DiffuseLight(const ColorRGB& c) : emit(make_shared<SolidColor>(c)) {}

  virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                       ColorRGB& attenuation, Ray& scattered) const override {
    return false;
  }
  virtual ColorRGB emitted(Real u, Real v, const Point3& p) const override {
    return emit->value(u, v, p);
  }

 public:
  shared_ptr<Texture> emit;
};
}  // namespace rt
}  // namespace dym
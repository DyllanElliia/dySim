/*
 * @Author: DyllanElliia
 * @Date: 2022-03-07 15:27:21
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-07 15:29:43
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
#include "render/randomFun.hpp"
namespace dym {
namespace rt {
class DiffuseLight : public Material {
public:
  DiffuseLight(shared_ptr<Texture> a) : emit(a) {}
  DiffuseLight(const ColorRGB &c) : emit(make_shared<SolidColor>(c)) {}

  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const override {
    return false;
  }
  virtual ColorRGB emitted(const Ray &r_in,
                           const HitRecord &rec) const override {
    return emit->value(rec.u, rec.v, rec.p);
  }
  virtual Ray gen_photon_r(const Point3 &p, const Vector3 &n) override {
    return Ray(p, random_in_hemisphere(n));
  }

public:
  shared_ptr<Texture> emit;
};
} // namespace rt
} // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2022-03-03 15:24:14
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-23 17:30:38
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"

// pdf
#include "../pdf/pdf.hpp"
namespace dym {
namespace rt {
class Lambertian : public Material {
public:
  Lambertian(const ColorRGB &a) : albedo(make_shared<SolidColor>(a)) {}
  Lambertian(const shared_ptr<Texture> &a) : albedo(a) {}

  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const override {
    // auto scatter_direction = rec.normal + random_unit_vector();

    // // Catch degenerate scatter direction
    // if (scatter_direction == 0.0) scatter_direction = rec.normal;

    // // scattered = Ray(rec.p, scatter_direction);
    // // attenuation = albedo->value(rec.u, rec.v, rec.p);

    // scattered = Ray(rec.p, scatter_direction.normalize(), r_in.time());
    // attenuation = albedo->value(rec.u, rec.v, rec.p);
    // pdf = rec.normal.dot(scattered.direction()) / pi;
    // return true;

    srec.is_specular = false;
    srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
    srec.pdf_ptr = make_shared<cosine_pdf>(rec.normal);
    return true;
  }

  virtual Vector3 BRDF_Evaluate(const Ray &r_in, const Ray &scattered,
                                const HitRecord &rec,
                                const ScatterRecord &srec) const {
    auto cosine = rec.normal.dot(scattered.direction().normalize());
    return cosine < 0 ? 0 : srec.attenuation * cosine / pi;
  }

public:
  shared_ptr<Texture> albedo;
};
} // namespace rt
} // namespace dym
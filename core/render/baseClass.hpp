/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:58:59
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-04 15:58:22
 * @Description:
 */
#pragma once
#include "ray.hpp"
#include "BVH/aabb.hpp"
#include "texture/solidColor.hpp"

namespace dym {
namespace rt {
class Hittable {
 public:
  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const = 0;
  virtual bool bounding_box(aabb& output_box) const = 0;
};

class Material {
 public:
  virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                       ColorRGB& attenuation, Ray& scattered) const = 0;
  virtual ColorRGB emitted(Real u, Real v, const Point3& p) const {
    return ColorRGB(0.f);
  }
};
}  // namespace rt
}  // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:58:59
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-03 15:19:14
 * @Description:
 */
#pragma once
#include "ray.hpp"

namespace dym {
namespace rt {
class Hittable {
 public:
  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const = 0;
};

class Material {
 public:
  virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                       ColorRGB& attenuation, Ray& scattered) const = 0;
};
}  // namespace rt
}  // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:58:59
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-01 15:06:30
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
}  // namespace rt
}  // namespace dym
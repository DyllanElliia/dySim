/*
 * @Author: DyllanElliia
 * @Date: 2022-03-04 15:19:01
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-04 15:40:03
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
namespace dym {
namespace rt {
class Texture {
 public:
  virtual ColorRGB value(const Real &u, const Real &v,
                         const Point3 &p) const = 0;
};
}  // namespace rt
}  // namespace dym
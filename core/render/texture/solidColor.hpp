/*
 * @Author: DyllanElliia
 * @Date: 2022-03-04 15:39:42
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-04 16:04:06
 * @Description:
 */
#pragma once
#include "texture.hpp"
namespace dym {
namespace rt {
class SolidColor : public Texture {
 public:
  SolidColor() {}

  SolidColor(const Real &red, const Real &green, const Real &blue)
      : color_value(ColorRGB({red, green, blue})) {}

  SolidColor(const ColorRGB &color) : color_value(color) {}

  virtual ColorRGB value(const Real &u, const Real &v,
                         const Point3 &p) const override {
    return color_value;
  }

 private:
  ColorRGB color_value;
};
}  // namespace rt
}  // namespace dym
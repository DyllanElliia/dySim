/*
 * @Author: DyllanElliia
 * @Date: 2022-03-07 15:09:39
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-11 14:25:03
 * @Description:
 */
#pragma once
#include "texture.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#endif

namespace dym {
namespace rt {
class ImageTexture : public Texture {
 public:
  const static int bytes_per_pixel = 3;

  ImageTexture() : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

  ImageTexture(const char* filename, const Real& lightIntensity = 1.f)
      : lightIntensity(lightIntensity / 255.f) {
    auto components_per_pixel = bytes_per_pixel;
    data = stbi_load(filename, &width, &height, &components_per_pixel,
                     components_per_pixel);
    if (!data) {
      DYM_ERROR("rt.texture error: Could not load texture image file '" +
                std::string(filename) + "'.");
      width = height = 0;
    }

    bytes_per_scanline = bytes_per_pixel * width;
    qprint(width, height);
  }

  ~ImageTexture() { delete data; }

  virtual ColorRGB value(Real u, Real v, const Vector3& p) const override {
    // If we have no texture data, then return solid cyan as a debugging aid.
    if (data == nullptr) return ColorRGB(0.f);

    // Clamp input texture coordinates to [0,1] x [1,0]
    auto ou = u, ov = v;
    u = clamp(u, 0.0, 1.0);
    v = 1.0 - clamp(v, 0.0, 1.0);  // Flip V to image coordinates
    // qprint("it1");
    auto i = static_cast<int>(u * width);
    auto j = static_cast<int>(v * height);
    // qprint("it2");

    // Clamp integer mapping, since actual coordinates should be less than 1.0
    if (i >= width) i = width - 1;
    if (j >= height) j = height - 1;
    // qprint("it3");

    auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;
    if (j < 0 || j >= height || i < 0 || i >= width)
      qprint("it4", ou, ov, u, v, i, j);

    return ColorRGB({Real(pixel[0]), Real(pixel[1]), Real(pixel[2])}) *
           lightIntensity;
  }

 private:
  unsigned char* data;
  int width, height;
  int bytes_per_scanline;
  Real lightIntensity;

;
};
}  // namespace rt
}  // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2022-03-07 15:09:39
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-29 14:32:48
 * @Description:
 */
#pragma once
#include "render/randomFun.hpp"
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
template <int bytes_per_pixel = 3> class ImageTexture : public Texture {
public:
  ImageTexture() : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

  ImageTexture(
      const std::string filename, const Real &lightIntensity = 1.f,
      std::function<ColorRGB(ColorRGBA &)> colorProcessFun =
          [](ColorRGBA &c) -> ColorRGB { return c; })
      : lightIntensity(lightIntensity), colorProcess(colorProcessFun) {
    auto components_per_pixel = bytes_per_pixel;
    data = stbi_load(filename.c_str(), &width, &height, &components_per_pixel,
                     components_per_pixel);
    if (!data) {
      DYM_ERROR("rt.texture error: Could not load texture image file '" +
                filename + "'.");
      width = height = 0;
    }

    bytes_per_scanline = bytes_per_pixel * width;
    qprint(width, height, components_per_pixel);
  }

  ~ImageTexture() { delete data; }

  virtual ColorRGB value(Real u, Real v, const Vector3 &p) const override {
    // If we have no texture data, then return solid cyan as a debugging aid.
    if (data == nullptr)
      return ColorRGB(0.f);

    // Clamp input texture coordinates to [0,1] x [1,0]
    auto ou = u, ov = v;
    u = clamp(u, 0.0, 1.0);
    v = 1.0 - clamp(v, 0.0, 1.0); // Flip V to image coordinates
    // qprint("it1");
    auto i = static_cast<int>(u * width);
    auto j = static_cast<int>(v * height);
    // qprint("it2");

    // Clamp integer mapping, since actual coordinates should be less than 1.0
    if (i >= width)
      i = width - 1;
    if (j >= height)
      j = height - 1;
    // qprint("it3");

    auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;
    if (j < 0 || j >= height || i < 0 || i >= width)
      // qprint("it4", ou, ov, u, v, i, j);
      return ColorRGB(0.f);
    ColorRGBA res({0, 0, 0, 1});

    if (overSampling) {
      Real a2_i = Real(width) * Real(height);
      Real posu = u * width, posv = v * height;
      res = 0.;
      Loop<int, 2>([&](auto x) {
        Loop<int, 2>([&](auto y) {
          auto offset = pixel + x * bytes_per_scanline + y * bytes_per_pixel;
          if (offset >= data + height * bytes_per_scanline)
            offset = pixel;
          Real u = abs(Real(i + x - posu) * Real(j + y - posv));
          Loop<int, bytes_per_pixel>(
              [&](auto ii) { res[ii] += u * Real(offset[ii]) / 255.; });
        });
      });
    } else
      Loop<int, bytes_per_pixel>(
          [&](auto ii) { res[ii] = Real(pixel[ii]) / 255.; });
    return colorProcess(res) * lightIntensity;
  }

private:
  unsigned char *data;
  int bytes_per_scanline;
  Real lightIntensity;
  std::function<ColorRGB(ColorRGBA &)> colorProcess;

public:
  int width, height;
  bool overSampling = true;
  ;
};
} // namespace rt
} // namespace dym
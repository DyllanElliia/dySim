#include "dyMath.hpp"
#include "render/randomFun.hpp"
#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>
#include <memory>

using SolidColor = dym::rt::SolidColor;
using ImageTexture = dym::rt::ImageTexture<3>;
using ImageTexRGBA = dym::rt::ImageTexture<4>;
_DYM_FORCE_INLINE_ auto lambertianSur(dym::rt::ColorRGB color) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(color);
  dmat.subSurface = std::make_shared<SolidColor>(0.5);
  dmat.metallic = std::make_shared<SolidColor>(0.0);
  dmat.specular = std::make_shared<SolidColor>(.1);
  dmat.specularTint = std::make_shared<SolidColor>(.2);
  dmat.roughness = std::make_shared<SolidColor>(.5);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

_DYM_FORCE_INLINE_ auto jadeSur(dym::rt::ColorRGB color, Real fuzz) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(color);
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(0.4);
  dmat.specular = std::make_shared<SolidColor>(fuzz);
  dmat.specularTint = std::make_shared<SolidColor>(.2);
  dmat.roughness = std::make_shared<SolidColor>(.5);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.8);
  dmat.sheenTint = std::make_shared<SolidColor>(0.8);
  dmat.clearcoat = std::make_shared<SolidColor>(0.6);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.8);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

_DYM_FORCE_INLINE_ auto magicSur(std::shared_ptr<dym::rt::Texture> color) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = color;
  dmat.subSurface = std::make_shared<SolidColor>(0.5);
  dmat.metallic = std::make_shared<SolidColor>(0.0);
  dmat.specular = std::make_shared<SolidColor>(.1);
  dmat.specularTint = std::make_shared<SolidColor>(.2);
  dmat.roughness = std::make_shared<SolidColor>(.5);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

auto cornell_box() {
  dym::rt::HittableList objects;
  Real fuzz = 0.2;
  auto red = lambertianSur(dym::rt::ColorRGB({.65, .05, .05}));
  auto white = lambertianSur(dym::rt::ColorRGB({.73, .73, .73}));
  auto green = lambertianSur(dym::rt::ColorRGB({.12, .45, .15}));
  auto light = std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB(20));

  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 1, green));
  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 0, red));

  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 0, white));
  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 1, white));
  objects.add(std::make_shared<dym::rt::xy_rect>(0, 1, 0, 1, 1, white));

  //   Real begin = 0.35, end = 0.65;
  //   objects.add(
  //       std::make_shared<dym::rt::xz_rect>(begin, end, begin, end, 0.998,
  //       light));

  return dym::rt::BvhNode(objects);
}

auto cornell_box_magic() {
  dym::rt::HittableList objects;
  Real fuzz = 0.2;
  auto red = magicSur(std::make_shared<ImageTexRGBA>(
      "./image/magic/cirLevel2.png", 1., [](dym::rt::ColorRGBA &c) {
        return dym::lerp(dym::Vector3{.65, .05, .05}, dym::Vector3(c), c[3]);
        // return dym::rt::ColorRGB(c[3]);
      }));
  auto white = magicSur(std::make_shared<ImageTexRGBA>(
      "./image/magic/cirLevel2.png", 1., [](dym::rt::ColorRGBA &c) {
        return dym::lerp(dym::Vector3{.73, .73, .73}, dym::Vector3(c), c[3]);
      }));
  auto green = magicSur(std::make_shared<ImageTexRGBA>(
      "./image/magic/cirLevel2.png", 1., [](dym::rt::ColorRGBA &c) {
        return dym::lerp(dym::Vector3{.12, .45, .15}, dym::Vector3(c), c[3]);
      }));
  auto light = std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB(20));

  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 1, green));
  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 0, red));

  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 0, white));
  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 1, white));
  objects.add(std::make_shared<dym::rt::xy_rect>(0, 1, 0, 1, 1, white));

  Real begin = 0.35, end = 0.65;
  objects.add(
      std::make_shared<dym::rt::xz_rect>(begin, end, begin, end, 0.998, light));

  return dym::rt::BvhNode(objects);
}
namespace dym {
namespace rt {
class magicLight : public Material {
public:
  magicLight(std::string &path, Real lightIntensity) {
    mask = std::make_shared<ImageTexRGBA>(path, 1., [](dym::rt::ColorRGBA &c) {
      return dym::rt::ColorRGB(c[3]);
    });
    emit = std::make_shared<SolidColor>(dym::rt::ColorRGB(1.));
  }

  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const override {
    return false;
    // return random_real() > mask->value(rec.u, rec.v, rec.p)[0] ? false :
    // true;
    srec.is_specular = true;
    srec.pdf_ptr = nullptr;
    srec.specular_ray = Ray(rec.p, r_in.direction().normalize(), r_in.time());
    srec.is_light = true;
    // return true;
    return random_real() > mask->value(rec.u, rec.v, rec.p)[0];
  }
  virtual ColorRGB emitted(const Ray &r_in, const HitRecord &rec, Real u,
                           Real v, const Point3 &p) const override {
    // return emit->value(u, v, p);
    return emit->value(u, v, p) * mask->value(rec.u, rec.v, rec.p)[0];
  }

public:
  shared_ptr<Texture> emit;
  shared_ptr<Texture> mask;
};
} // namespace rt
} // namespace dym
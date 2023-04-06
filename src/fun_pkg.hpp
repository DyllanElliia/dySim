#include "dyMath.hpp"
#include "render/material/dielectric.hpp"
#include "render/randomFun.hpp"
#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>
#include <memory>
#include <tuple>

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

_DYM_FORCE_INLINE_ auto metalSur(dym::rt::ColorRGB color, Real fuzz) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(color);
  dmat.subSurface = std::make_shared<SolidColor>(.0);
  dmat.metallic = std::make_shared<SolidColor>(.8);
  dmat.specular = std::make_shared<SolidColor>(.9);
  dmat.specularTint = std::make_shared<SolidColor>(.4);
  dmat.roughness = std::make_shared<SolidColor>(fuzz);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.8);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.8);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

_DYM_FORCE_INLINE_ auto glassSur(dym::rt::ColorRGB color, Real ref) {
  auto die_surface = std::make_shared<dym::rt::Dielectric>(color, ref);

  return die_surface;
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

_DYM_FORCE_INLINE_ auto magicSur(dym::rt::ColorRGB color,
                                 dym::rt::ColorRGB light, Real fuzz) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(color);
  dmat.subSurface = std::make_shared<SolidColor>(.0);
  dmat.metallic = std::make_shared<SolidColor>(.8);
  dmat.specular = std::make_shared<SolidColor>(.9);
  dmat.specularTint = std::make_shared<SolidColor>(.4);
  dmat.roughness = std::make_shared<SolidColor>(fuzz);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.8);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.8);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);
  dmat.lightEmit = std::make_shared<SolidColor>(light);

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

  objects.add(std::make_shared<dym::rt::yz_rect<true>>(0, 1, 0, 1, 1, green));
  objects.add(std::make_shared<dym::rt::yz_rect<true>>(0, 1, 0, 1, 0, red));

  objects.add(std::make_shared<dym::rt::xz_rect<true>>(0, 1, 0, 1, 0, white));
  objects.add(std::make_shared<dym::rt::xz_rect<true>>(0, 1, 0, 1, 1, white));
  objects.add(std::make_shared<dym::rt::xy_rect<true>>(0, 1, 0, 1, 1, white));

  Real begin = 0.35, end = 0.65;
  objects.add(std::make_shared<dym::rt::xz_rect<true>>(begin, end, begin, end,
                                                       0.998, light));

  return dym::rt::BvhNode(objects);
}

auto cornell_box3() {
  dym::rt::HittableList objects;
  dym::rt::HittableList lights;
  Real fuzz = 0.2;
  auto red = lambertianSur(dym::rt::ColorRGB({.65, .05, .05}));
  auto white = lambertianSur(dym::rt::ColorRGB({.73, .73, .73}));
  auto green = lambertianSur(dym::rt::ColorRGB({.12, .45, .15}));
  auto light = std::make_shared<dym::rt::DiffuseLight>(
      40. * dym::rt::ColorRGB{0.8, 0.6, 0.5});

  Real begin = 0.49, end = 0.51;
  Real greenHeiOff = -0.4;
  // objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 1, green));
  // objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 0, red));
  objects.addObject<dym::rt::Box>(dym::Vector3{-0.3, 0, 0},
                                  dym::Vector3{0, 1, begin}, white);
  objects.addObject<dym::rt::Box>(dym::Vector3{-0.3, 0, end},
                                  dym::Vector3{0, 1, 1.0}, white);
  objects.addObject<dym::rt::Box>(dym::Vector3{1.0, 0, 0},
                                  dym::Vector3{1.5, begin + greenHeiOff, 1},
                                  white);
  objects.addObject<dym::rt::Box>(dym::Vector3{1.0, end + greenHeiOff, 0},
                                  dym::Vector3{1.5, 1, 1}, white);

  objects.add(
      std::make_shared<dym::rt::xz_rect<true>>(-.1, 2.1, -.1, 1.1, 0, white));
  objects.add(
      std::make_shared<dym::rt::xz_rect<true>>(-.1, 2.1, -.1, 1.1, 1, white));
  objects.add(
      std::make_shared<dym::rt::xy_rect<true>>(-.1, 2.1, -.1, 1.1, 1, white));

  auto lightobj =
      std::make_shared<dym::rt::yz_rect<true>>(0, 1, begin, end, -.2, light);
  objects.add(lightobj);
  lights.add(lightobj);
  lightobj = std::make_shared<dym::rt::yz_rect<true>>(
      begin + greenHeiOff, end + greenHeiOff, 0, 1, 1.1, light);
  objects.add(lightobj);
  lights.add(lightobj);

  return std::make_tuple(dym::rt::BvhNode(objects), lights);
}

namespace dym {
namespace rt {
class magicLight : public Material {
public:
  magicLight(std::string &path, Real lightIntensity) {
    auto tex =
        std::make_shared<ImageTexRGBA>(path, 1., [](dym::rt::ColorRGBA &c) {
          return dym::rt::ColorRGB(c[3]);
        });
    mask = tex;
    tex->overSampling = false;
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
    // return true;
    return random_real() > mask->value(rec.u, rec.v, rec.p)[0];
  }
  virtual ColorRGB emitted(const Ray &r_in,
                           const HitRecord &rec) const override {
    return emit->value(rec.u, rec.v, rec.p) * mask->value(rec.u, rec.v, rec.p);
  }

public:
  shared_ptr<Texture> emit;
  shared_ptr<Texture> mask;
};

class Mask : public Hittable {
public:
  Mask(const std::string &path, shared_ptr<Hittable> obj) : obj(obj) {
    auto texptr =
        std::make_shared<ImageTexRGBA>(path, 1., [](dym::rt::ColorRGBA &c) {
          return dym::rt::ColorRGB(c[3]);
        });
    mask = texptr;
    // texptr->overSampling = false;
    // Real cnt = 0;
    // auto &width = texptr->width, height = texptr->height;
    // Vector3 p;
    // for (int i = 0; i < width; ++i)
    //   for (int j = 0; j < height; ++j)
    //     cnt += texptr->value(i / Real(width), j / Real(height), p)[0];
    // area = cnt / Real(width * height);
    // texptr->overSampling = true;
    // qprint(area);
  }

  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const override {
    HitRecord rec_;
    if (obj->hit(r, t_min, t_max, rec_)) {
      if (mask->value(rec_.u, rec_.v, rec_.p)[0] > 0) {
        rec = rec_;
        return true;
      } else
        return false;
    } else
      return false;
  }

  virtual bool bounding_box(aabb &output_box) const override {

    return obj->bounding_box(output_box);
  }

  virtual Real pdf_value(const Point3 &origin,
                         const Vector3 &v) const override {
    return obj->pdf_value(origin, v);
  }
  virtual Vector3 random(const Point3 &origin) const override {
    while (true) {
      auto v = obj->random(origin);
      Ray r(origin, v);
      HitRecord rec;
      if (obj->hit(r, 1e-7, 1e7, rec)) {
        if (random_real() > mask->value(rec.u, rec.v, rec.p)[0])
          return v;
      }
    }
  }

public:
  shared_ptr<Texture> mask;
  shared_ptr<Hittable> obj;
  Real area;
};
} // namespace rt
} // namespace dym
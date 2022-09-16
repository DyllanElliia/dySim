/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:34:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-19 13:50:24
 * @Description:
 */
#include "render/object/sphere.hpp"
#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>

_DYM_FORCE_INLINE_ auto whiteMetalSur(Real objcolor, Real fuzz = 0) {
  auto white_surface =
      std::make_shared<dym::rt::Metal>(dym::rt::ColorRGB(objcolor), fuzz);

  return white_surface;
}

using SolidColor = dym::rt::SolidColor;
using ImageTexture = dym::rt::ImageTexture;
_DYM_FORCE_INLINE_ auto disneryBRDF(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3{0.9, 0.75, 0.2});
  dmat.subSurface = std::make_shared<SolidColor>(0.1);
  dmat.metallic = std::make_shared<SolidColor>(dym::min(0.95, x));
  dmat.specular = std::make_shared<SolidColor>(.9);
  dmat.specularTint = std::make_shared<SolidColor>(.2);
  dmat.roughness = std::make_shared<SolidColor>(.1);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

_DYM_FORCE_INLINE_ auto lightEarthSur() {
  auto earth_texture =
      std::make_shared<dym::rt::ImageTexture>("image/earthmap.jpg", 3);
  auto earth_surface = std::make_shared<dym::rt::DiffuseLight>(earth_texture);

  return earth_surface;
}
auto cornell_box() {
  dym::rt::HittableList objects;
  Real fuzz = 0.2;
  auto red =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({.65, .05, .05}));
  auto white =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({.73, .73, .73}));
  auto green =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({.12, .45, .15}));
  auto light = std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB(20));

  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 1, green));
  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 0, red));

  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 0, white));
  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 1, white));
  objects.add(std::make_shared<dym::rt::xy_rect>(0, 1, 0, 1, 1, white));
  // objects.add(std::make_shared<dym::rt::xy_rect>(0, 1, 0, 1, 0, white));

  Real begin = 0.35, end = 0.65;
  objects.add(
      std::make_shared<dym::rt::xz_rect>(begin, end, begin, end, 0.998, light));

  return dym::rt::BvhNode(objects);
}

int main(int argc, char const *argv[]) {
  const auto aspect_ratio = 2000. / 300.;
  const int image_width = 2000;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  int samples_per_pixel = 10;
  const int max_depth = 50;
  dym::Tensor<dym::Vector<Real, dym::PIC_RGB>> image(
      0, dym::gi(image_height, image_width));
  dym::Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> imageP(
      0, dym::gi(image_height, image_width));
  // World

  dym::rt::HittableList world;
  dym::rt::HittableList lights;
  Real begin = 0.35, end = 0.65;
  lights.add(std::make_shared<dym::rt::xz_rect>(
      begin, end, begin, end, 0.998, std::shared_ptr<dym::rt::Material>()));

  world.add(std::make_shared<dym::rt::BvhNode>(cornell_box()));

  for (Real x = 0.1; x < 0.9; x += 0.08)
  // for (Real y = 0.1; y < 0.9; y += 0.08 * 0.5)
  {
    world.addObject<dym::rt::Sphere>(dym::rt::Point3({x, 0.03, 0.5}), 0.03,
                                     disneryBRDF((x - 0.1) / 0.8));
    // world.addObject<dym::rt::Sphere>(
    //     dym::rt::Point3({x, 0.15, y}), 0.1,
    //     whiteMetalSur(1 - y, lerp(0.0, 0.05, (x - 0.15))));
    // qprint(x, y);
  }

  dym::rt::RtRender render(image_width, image_height);

  // Camera
  dym::rt::Point3 lookfrom({0.5, .2, 0.});
  dym::rt::Point3 lookat({0.5, 0.03, 0.5});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.0;
  render.cam.setCamera(lookfrom, lookat, vup, 15, aspect_ratio, aperture,
                       dist_to_focus);

  // dym::rt::Point3 lookfrom({0.5, 0.5, -1.35});
  // dym::rt::Point3 lookat({0.5, 0.5, 0});
  // dym::Vector3 vup({0, 1, 0});
  // auto dist_to_focus = (lookfrom - lookat).length();
  // auto aperture = 2.0;
  // render.cam.setCamera(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
  //                      dist_to_focus);

  render.worlds.addObject<dym::rt::BvhNode>(world);
  render.lights = lights;

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  Real t = 0.4, t_inv = 1 - t;
  dym::TimeLog time;
  int ccc = 1;

  time.reStart();
  gui.update([&]() {
    dym::TimeLog partTime;
    render.render(samples_per_pixel, max_depth);
    render.denoise();
    ccc++;
    time.record();
    time.reStart();
    auto image = render.getFrame();
    dym::imwrite(image, "./rt_out/sample/disnery/frame_" +
                            std::to_string(ccc - 1) + ".jpg");
    gui.imshow(image);
  });
  return 0;
}

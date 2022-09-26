/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:34:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-19 13:50:24
 * @Description:
 */
#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>
#include <memory>
_DYM_FORCE_INLINE_ auto earthSur() {
  auto earth_texture =
      std::make_shared<dym::rt::ImageTexture<3>>("image/earthmap.jpg");
  auto earth_surface = std::make_shared<dym::rt::Lambertian>(earth_texture);

  return earth_surface;
}
_DYM_FORCE_INLINE_ auto whiteSur() {
  auto white_texture =
      std::make_shared<dym::rt::SolidColor>(dym::rt::ColorRGB(0.8f));
  auto white_surface = std::make_shared<dym::rt::Lambertian>(white_texture);

  return white_surface;
}
_DYM_FORCE_INLINE_ auto whiteMetalSur(Real fuzz = 0) {
  auto white_surface =
      std::make_shared<dym::rt::Metal>(dym::rt::ColorRGB(1.0f), fuzz);

  return white_surface;
}
_DYM_FORCE_INLINE_ auto whiteGalssSur() {
  auto white_surface = std::make_shared<dym::rt::Dielectric>(1.5);

  return white_surface;
}

_DYM_FORCE_INLINE_ auto blueConSur() {
  auto blue_surface = std::make_shared<dym::rt::Dielectric>(1.5);

  return blue_surface;
}

_DYM_FORCE_INLINE_ auto lightEarthSur() {
  auto earth_texture =
      std::make_shared<dym::rt::ImageTexture<3>>("image/earthmap.jpg", 3);
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

  Real begin = 0.35, end = 0.65;
  objects.add(
      std::make_shared<dym::rt::xz_rect>(begin, end, begin, end, 0.998, light));

  return dym::rt::BvhNode(objects);
}

int main(int argc, char const *argv[]) {
  const auto aspect_ratio = 1.f;
  const int image_width = 800;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 3;
  const int max_depth = 20;
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

  auto boxo = std::make_shared<dym::rt::Box>(
      dym::rt::Point3(-1), dym::rt::Point3(1), whiteGalssSur());

  dym::Matrix3 scalem1 = dym::matrix::identity<Real, 4>(0.08);
  dym::Matrix3 scalem2 = dym::matrix::identity<Real, 4>(0.1);
  scalem2[1][1] = 0.2;

  dym::Vector3 translate0({0.5, -0.25, 0.5});
  dym::Vector3 translate1({0.5, 0.55, 0.5});
  dym::Vector3 translate2({0.5, 0.2, 0.5});

  dym::Quaternion rotate0 = dym::getQuaternion<Real>(0, {0, 1, 0});
  dym::Quaternion rotate1 =
      dym::getQuaternion<Real>(dym::atan(dym::sqrt(2.0)), {1, 0, 1});
  dym::Quaternion rotate2 = dym::getQuaternion<Real>(dym::Pi / 4, {0, 1, 0});

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::Box>(dym::rt::Point3(-0.3),
                                     dym::rt::Point3(0.3), whiteSur()),
      rotate2.to_matrix(), translate0));

  world.add(std::make_shared<dym::rt::Transform3>(
      boxo, scalem1 * rotate1.to_matrix(), translate1));

  world.add(std::make_shared<dym::rt::ConstantMedium>(
      std::make_shared<dym::rt::Transform3>(boxo, scalem2 * rotate2.to_matrix(),
                                            translate2),
      200, dym::rt::ColorRGB({0.2, 0.4, 0.9})));

  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.8, 0.2, 0.8}),
                                              0.1, earthSur()));

  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.2, 0.2, 0.2}),
                                              0.1, whiteMetalSur(0.1)));
  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.8, 0.2, 0.2}),
                                              0.1, whiteGalssSur()));

  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.2, 0.2, 0.8}),
                                              0.1, whiteMetalSur()));

  auto mat = earthSur();
  auto mat2 = whiteMetalSur(0.2);
  dym::Vector3 tnormal({0, 0, -1});
  dym::rt::Vertex v0(dym::rt::Point3({0, 0, 0.999}), tnormal, 0, 0),
      v1(dym::rt::Point3({0, 1, 0.999}), tnormal, 0, 1),
      v2(dym::rt::Point3({1, 1, 0.999}), tnormal, 1, 1);
  dym::rt::Vertex v3(dym::rt::Point3({0, 0, 0.999}), tnormal, 0, 0),
      v4(dym::rt::Point3({1, 0, 0.999}), tnormal, 1, 0),
      v5(dym::rt::Point3({1, 1, 0.999}), tnormal, 1, 1);
  world.add(std::make_shared<dym::rt::Triangle>(v0, v1, v2, mat));
  world.add(std::make_shared<dym::rt::Triangle>(v3, v4, v5, mat2));

  // world.add(std::make_shared<dym::rt::xy_rect>(0, 1, 0, 1, 1, mat));

  //   auto worlds = dym::rt::BvhNode(world);

  // Camera
  dym::rt::Point3 lookfrom({0.5, 0.5, -1.35});
  dym::rt::Point3 lookat({0.5, 0.5, 0});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.0;

  dym::rt::RtRender render(image_width, image_height);

  render.cam.setCamera(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
                       dist_to_focus);

  render.worlds.addObject<dym::rt::BvhNode>(world);
  render.lights = lights;

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  Real t = 0.4, t_inv = 1 - t;
  dym::TimeLog time;
  int ccc = 1;

  // for (int i = 0; i < 5; ++i) {
  //   auto test =
  //       std::make_shared<dym::rt::cosine_dir_pdf>(dym::Vector3{0, 1, 0},
  //       0.1);
  //   auto v = test->generate();
  //   qprint(v, test->value(v));
  // }

  time.reStart();
  gui.update([&]() {
    // lookfrom = dym::rt::Point3(
    //     {0.5 + dym::cos(ccc * 0.02 / 3.1415926585), 0.5, -1.35});
    // render.cam.setCamera(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
    //                      dist_to_focus);
    dym::TimeLog partTime;
    render.render(samples_per_pixel, max_depth, [](const dym::rt::Ray &r) {
      dym::Vector3 unit_direction = r.direction().normalize();
      Real t = 0.5f * (unit_direction.y() + 1.f);
      return (1.f - t) * dym::rt::ColorRGB(1.f) +
             t * dym::rt::ColorRGB({0.5f, 0.7f, 1.0f});
    });

    qprint("fin render part time:", partTime.getRecord());
    partTime.reStart();

    render.denoise();

    qprint("fin denoise part time:", partTime.getRecord());
    partTime.reStart();

    ccc++;
    time.record();
    time.reStart();
    // auto image = render.getFrameGBuffer("depth", 100);
    auto image = render.getFrame();
    dym::imwrite(image,
                 "./rt_out/sample/4/frame_" + std::to_string(ccc - 1) + ".jpg");
    gui.imshow(image);
  });
  return 0;
}

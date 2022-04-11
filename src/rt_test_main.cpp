/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:34:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-11 17:17:08
 * @Description:
 */
#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>
_DYM_FORCE_INLINE_ auto earthSur() {
  auto earth_texture =
      std::make_shared<dym::rt::ImageTexture>("image/earthmap.jpg");
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
      std::make_shared<dym::rt::Metal>(dym::rt::ColorRGB(0.8f), fuzz);

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
  auto light =
      std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB({5, 5, 5}));

  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 1, green));
  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 0, red));
  objects.add(
      std::make_shared<dym::rt::xz_rect>(0.2, 0.8, 0.2, 0.8, 0.99, light));
  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 0, white));
  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 1, white));
  objects.add(std::make_shared<dym::rt::xy_rect>(0, 1, 0, 1, 1, white));

  return dym::rt::BvhNode(objects);
}

auto cornell_box2() {
  dym::rt::HittableList objects;
  Real fuzz = 0.2;
  auto red =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({.65, .05, .05}));
  auto white =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({.73, .73, .73}));
  auto green =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({.12, .45, .15}));
  auto light = std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB(15));

  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 1, green));
  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 0, red));

  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 0, white));

  objects.add(std::make_shared<dym::rt::xy_rect>(0, 1, 0, 1, 1, white));

  Real begin = 0.35, end = 0.65;
  objects.add(std::make_shared<dym::rt::xz_rect>(0, end, 0, begin, 1, white));
  objects.add(std::make_shared<dym::rt::xz_rect>(end, 1, 0, end, 1, white));
  objects.add(std::make_shared<dym::rt::xz_rect>(begin, 1, end, 1, 1, white));
  objects.add(std::make_shared<dym::rt::xz_rect>(0, begin, begin, 1, 1, white));
  objects.add(
      std::make_shared<dym::rt::xz_rect>(begin, end, begin, end, 0.998, light));

  // auto glass = whiteGalssSur();
  // objects.add(
  //     std::make_shared<dym::rt::Box>(dym::rt::Point3({begin, 0.90, begin}),
  //                                    dym::rt::Point3({end, 1.1, end}),
  //                                    glass));

  // objects.add(std::make_shared<dym::rt::ConstantMedium>(
  //     std::make_shared<dym::rt::Sphere>(dym::rt::Point3(0.5), 2, glass),
  //     0.001, dym::rt::ColorRGB(1)));
  return dym::rt::BvhNode(objects);
}

int main(int argc, char const* argv[]) {
  qprint(dym::rt::random_cosine_direction());
  qprint(std::exp(800),
         exp(800) == exp(800) ? "inf is same" : "inf is not same");
  // const auto aspect_ratio = 16.0 / 9.0;
  const auto aspect_ratio = 1.f;
  const int image_width = 600;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 5;
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
  // lights.add(make_shared<dym::rt::Sphere>(
  //     dym::rt::Point3(0.5), 1, std::shared_ptr<dym::rt::Material>()));

  world.add(std::make_shared<dym::rt::BvhNode>(cornell_box2()));
  // world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.5, 0.5,
  // 0.5}),
  //                                             0.2, earthSur()));

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

  // qprint(rotate1, rotate2, dym::atan(dym::sqrt(2.0)), dym::sqrt(2.0));

  // qprint(translate1 * scalem1);

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::Box>(dym::rt::Point3(-0.3),
                                     dym::rt::Point3(0.3), whiteSur()),
      rotate2.to_matrix(), translate0));

  // world.add(sphereo);
  world.add(std::make_shared<dym::rt::Transform3>(
      boxo, scalem1 * rotate1.to_matrix(), translate1));

  lights.add(std::make_shared<dym::rt::Transform3>(
      boxo, scalem1 * rotate1.to_matrix(), translate1));

  world.add(std::make_shared<dym::rt::ConstantMedium>(
      std::make_shared<dym::rt::Transform3>(boxo, scalem2 * rotate2.to_matrix(),
                                            translate2),
      200, dym::rt::ColorRGB({0.2, 0.4, 0.9})));
  // world.add(std::make_shared<dym::rt::Transform3>(
  //     boxo, scalem2 * rotate2.to_matrix(), translate2));

  // world.add(std::make_shared<dym::rt::Box>(dym::rt::Point3({0.7, 0.0,
  // 0.1}),
  //                                          dym::rt::Point3({1.0, 0.1,
  //                                          0.8}), whiteMetalSur()));

  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.8, 0.2, 0.8}),
                                              0.1, earthSur()));

  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.2, 0.2, 0.2}),
                                              0.1, whiteMetalSur(0.8)));
  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.8, 0.2, 0.2}),
                                              0.1, whiteGalssSur()));

  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.2, 0.2, 0.8}),
                                              0.1, whiteMetalSur()));

  // lights.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.2, 0.2,
  // 0.8}),
  //                                              0.1, whiteGalssSur()));
  auto mat = earthSur();
  auto mat2 = whiteMetalSur(0.8);
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

  auto worlds = dym::rt::BvhNode(world);

  // Camera
  dym::rt::Point3 lookfrom({0.5, 0.5, -1.35});
  dym::rt::Point3 lookat({0.5, 0.5, 0});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.0;

  dym::rt::Camera<false> cam(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
                             dist_to_focus);

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  Real t = 0.4, t_inv = 1 - t;
  dym::TimeLog time;
  int ccc = 1;

  time.reStart();
  gui.update([&]() {
    image.for_each_i([&](dym::Vector<Real, dym::PIC_RGB>& color, int i, int j) {
      auto oldColor = dym::sqr(color / 255.f) * Real(ccc - 1);
      color = 0.f;
      for (int samples = 0; samples < samples_per_pixel; samples++) {
        auto u = (Real)j / (image_width - 1);
        auto v = (Real)i / (image_height - 1);
        // auto u = (j + dym::rt::random_real()) / (image_width - 1);
        // auto v = (i + dym::rt::random_real()) / (image_height - 1);
        dym::rt::Ray r = cam.get_ray(u, v);
        // color += ray_color2(r, world, max_depth, [](const dym::rt::Ray& r)
        // {
        //   dym::Vector3 unit_direction = r.direction().normalize();
        //   Real t = 0.5f * (unit_direction.y() + 1.f);
        //   return (1.f - t) * dym::rt::ColorRGB(1.f) +
        //          t * dym::rt::ColorRGB({0.5f, 0.7f, 1.0f});
        // });
        // color += ray_color_pdf(r, worlds, nullptr, max_depth);
        color += ray_color_pdf(r, worlds,
                               std::make_shared<dym::rt::HittableList>(lights),
                               max_depth);
      }
      color = color * (1.f / Real(samples_per_pixel));
      color = (color + oldColor) / Real(ccc);
      color = dym::clamp(dym::sqrt(color) * 255.f, 0.0, 255.99);
      // if (color[0] != color[0]) color[0] = 0.0;
      // if (color[1] != color[1]) color[1] = 0.0;
      // if (color[2] != color[2]) color[2] = 0.0;
      dym::Loop<int, 3>([&](auto pi) {
        if (dym::isnan(color[pi])) color[pi] = 0;
        // if (dym::isinf(color[pi])) color[pi] = 255;
      });
    });
    ccc++;
    // time.record();
    // time.reStart();

    // image = dym::filter2D(image, dym::Matrix3(1.f / 9.f));
    imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB>& e, int i) {
      e = image[i].cast<dym::Pixel>();
    });

    gui.imshow(imageP);
    if (ccc == 2 || ccc == 6 || ccc == 21 || ccc == 41) {
      qprint(ccc, time.getRecord());
      dym::imwrite(image, "./rt_out/rt_test" + std::to_string(ccc) + ".jpg");
    }
  });
  return 0;
}

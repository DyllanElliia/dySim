/*
 * @Author: DyllanElliia
 * @Date: 2022-04-12 16:21:12
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-14 14:55:36
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
  auto light = std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB(15));

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
  const int samples_per_pixel = 1;
  const int max_depth = 20;
  dym::Tensor<dym::Vector<Real, dym::PIC_RGB>> image(
      0, dym::gi(image_height, image_width));
  dym::Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> imageP(
      0, dym::gi(image_height, image_width));
  // World

  dym::rt::HittableList world;
  dym::rt::HittableList lights;

  Real begin = 0.35, end = 0.65;

  world.add(std::make_shared<dym::rt::BvhNode>(cornell_box()));
  lights.add(std::make_shared<dym::rt::xz_rect>(
      begin, end, begin, end, 0.998, std::shared_ptr<dym::rt::Material>()));

  std::vector<dym::rt::Point3> positions = {
      dym::rt::Point3({begin, begin, begin}),
      dym::rt::Point3({end, begin, begin}),
      dym::rt::Point3({begin, end, begin}),
      dym::rt::Point3({end, end, begin}),
      dym::rt::Point3({begin, begin, end}),
      dym::rt::Point3({end, begin, end}),
      dym::rt::Point3({begin, end, end}),
      dym::rt::Point3({end, end, end})};
  for (auto& p : positions) p -= 0.5;
  std::vector<dym::Vector3ui> faces = {
      dym::Vector3ui({0, 3, 1}), dym::Vector3ui({0, 2, 3}),
      dym::Vector3ui({0, 6, 2}), dym::Vector3ui({0, 4, 6}),
      dym::Vector3ui({2, 6, 3}), dym::Vector3ui({3, 6, 7}),
      dym::Vector3ui({0, 1, 5}), dym::Vector3ui({0, 5, 4}),
      dym::Vector3ui({1, 3, 5}), dym::Vector3ui({5, 3, 7}),
      dym::Vector3ui({5, 7, 6}), dym::Vector3ui({5, 6, 4})};

  dym::Quaternion rotate = dym::getQuaternion<Real>(dym::Pi, {0, 1, 0});

  dym::Quaternion rotate2 = dym::getQuaternion<Real>(dym::Pi / 4, {1, 0, 0});
  dym::Matrix3 scalem = dym::matrix::identity<Real, 3>(3.5);

  dym::Vector3 translation({0.4, 0, 0.5});

  // world.add(std::make_shared<dym::rt::Transform3>(
  //     std::make_shared<dym::rt::Mesh>(positions, faces, whiteSur()),
  //     scalem * rotate2.to_matrix() * rotate.to_matrix(), 0.5));

  dym::Model loader("./PLYFiles/ply/Bunny10K.ply");

  dym::TimeLog ttt;

  auto pAABB = [&](dym::rt::Hittable& obj) {
    dym::rt::aabb objaabb;
    obj.bounding_box(objaabb);
    qprint(objaabb.min(), objaabb.max());
    dym::rt::Ray r(0.5, dym::Vector3({0, 0, 1}));
    qprint("hit:", objaabb.hit(r, 0, 100000));
  };

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::Mesh>(loader.meshes[0], whiteMetalSur()),
      scalem * rotate.to_matrix(), translation));

  ttt.record();

  auto worlds = dym::rt::BvhNode(world);

  pAABB(worlds);
  pAABB(*(worlds.right));
  pAABB(*(worlds.left));
  getchar();

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
        if (dym::isinf(color[pi])) color[pi] = oldColor[pi];
      });
    });
    ccc++;
    time.record();
    time.reStart();
    dym::imwrite(image, "./rt_out/rt_test" + std::to_string(ccc) + ".jpg");

    // image = dym::filter2D(image, dym::Matrix3(1.f / 9.f));
    imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB>& e, int i) {
      e = image[i].cast<dym::Pixel>();
    });

    gui.imshow(imageP);
    // if (ccc == 2 || ccc == 6 || ccc == 21 || ccc == 41) {
    //   // qprint(ccc, time.getRecord());
    //   dym::imwrite(image, "./rt_out/rt_test" + std::to_string(ccc) + ".jpg");
    // }
  });
  return 0;
}

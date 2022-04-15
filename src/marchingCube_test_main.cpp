/*
 * @Author: DyllanElliia
 * @Date: 2022-04-15 15:13:05
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-15 17:26:31
 * @Description:
 */
#define DYM_USE_MARCHING_CUBES
#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>
#include <dySimulator.hpp>
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

int main(int argc, char const *argv[]) {
  dym::MLSMPM<dym::MidGrid, dym::OneStickyOtherSeparate> sim;
  sim.globalForce = dym::Vector3({0.f, -9.8 * 2.f, 0.f});
  std::default_random_engine re;
  std::uniform_real_distribution<Real> u(-1.f, 1.f);
  u_int n3 = 5000;
  dym::Tensor<dym::Vector3> newX(0, dym::gi(n3));

  newX.for_each_i([&](dym::Vector3 &pos) {
    pos = dym::Vector3({u(re), u(re), u(re)}) * 0.15f;
  });

  sim.addParticle(newX + dym::Vector3(0.5), sim.addLiquidMaterial());

  const Real dt = 1e-4;
  const int volume_n = 64;

  dym::Tensor<Real> volume(0, dym::gi(volume_n, volume_n, volume_n));
  auto Tp = [&](dym::Tensor<dym::Vector3> &x) {
    volume = 0;
    x.for_each_i([&](dym::Vector3 &pos, int i) {
      auto pos_off = pos * volume_n;
      auto pos_i = pos_off.cast<int>();
      volume[pos_i] = 1;
      dym::Loop<int, 2>([&](auto ii) {
        dym::Loop<int, 2>([&](auto jj) {
          dym::Loop<int, 2>([&](auto kk) {
            auto pos_ijk = pos_i + dym::Vector3i({ii, jj, kk});
            if (pos_ijk >= 0; pos_ijk < volume_n) volume[pos_ijk] = 1;
          });
        });
      });
    });
  };

  // Render
  const auto aspect_ratio = 1.f;
  const int image_width = 600;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 3;
  const int max_depth = 20;
  dym::Tensor<dym::Vector<Real, dym::PIC_RGB>> image(
      0, dym::gi(image_height, image_width));
  dym::Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> imageP(
      0, dym::gi(image_height, image_width));

  // static world
  dym::rt::HittableList world;
  dym::rt::HittableList lights;

  Real begin = 0.35, end = 0.65;

  world.add(std::make_shared<dym::rt::BvhNode>(cornell_box()));
  lights.add(std::make_shared<dym::rt::xz_rect>(
      begin, end, begin, end, 0.998, std::shared_ptr<dym::rt::Material>()));

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
  Real t = 1, t_inv = 1 - t;
  dym::TimeLog time;
  int ccc = 1;
  const int steps = 25;
  dym::Matrix3 scalem = dym::matrix::identity<Real, 3>(1 / Real(volume_n - 1));

  time.reStart();
  gui.update([&]() {
    Tp(sim.getPos());
    for (int i = 0; i < steps; ++i) sim.advance(dt);
    auto mesh = dym::marchingCubes(volume, 0.5);
    dym::rt::HittableList worlds;
    worlds.add(std::make_shared<dym::rt::BvhNode>(world));
    worlds.add(std::make_shared<dym::rt::Transform3>(
        std::make_shared<dym::rt::Mesh>(mesh, whiteMetalSur()), scalem));

    image.for_each_i([&](dym::Vector<Real, dym::PIC_RGB> &color, int i, int j) {
      auto color_pre = color;
      color = 0.f;
      for (int samples = 0; samples < samples_per_pixel; samples++) {
        auto u = (Real)j / (image_width - 1);
        auto v = (Real)i / (image_height - 1);
        dym::rt::Ray r = cam.get_ray(u, v);
        color += ray_color_pdf(r, worlds,
                               std::make_shared<dym::rt::HittableList>(lights),
                               max_depth);
      }
      color = color * (1.f / Real(samples_per_pixel));
      color = dym::clamp(dym::sqrt(color) * 255.f, 0.0, 255.99);
      color = t * color + t_inv * color_pre;
      dym::Loop<int, 3>([&](auto pi) {
        if (dym::isnan(color[pi])) color[pi] = 0;
        if (dym::isinf(color[pi])) color[pi] = color_pre[pi];
      });
    });
    t = 0.4;
    t_inv = 1 - t;
    ccc++;
    time.record();
    time.reStart();
    dym::imwrite(image,
                 "./rt_out/mctest/frame_" + std::to_string(ccc - 1) + ".png");

    // image = dym::filter2D(image, dym::Matrix3(1.f / 9.f));
    imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB> &e, int i) {
      e = image[i].cast<dym::Pixel>();
    });

    gui.imshow(imageP);
  });

  return 0;
}

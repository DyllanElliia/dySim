/*
 * @Author: DyllanElliia
 * @Date: 2022-04-15 15:13:05
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-25 15:20:20
 * @Description:
 */
#define DYM_USE_MARCHING_CUBES
#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>
#include <dySimulator.hpp>
_DYM_FORCE_INLINE_ auto earthSur() {
  auto earth_texture =
      std::make_shared<dym::rt::ImageTexture<3>>("image/earthmap.jpg");
  auto earth_surface = std::make_shared<dym::rt::Lambertian>(earth_texture);

  return earth_surface;
}
_DYM_FORCE_INLINE_ auto whiteSur() {
  auto white_texture =
      std::make_shared<dym::rt::SolidColor>(dym::rt::ColorRGB(1.0));
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

_DYM_FORCE_INLINE_ auto whiteWaterSur() {
  auto white_texture =
      std::make_shared<dym::rt::SolidColor>(dym::rt::ColorRGB({0.9, 1, 1}));
  auto white_surface =
      std::make_shared<dym::rt::Dielectric>(white_texture, 1.5);

  return white_surface;
}

_DYM_FORCE_INLINE_ auto whiteJullySur() {
  auto white_texture = std::make_shared<dym::rt::SolidColor>(
      dym::rt::ColorRGB({0.92, 0.33, 0.23}));
  auto white_surface =
      std::make_shared<dym::rt::Dielectric>(white_texture, 1.5);

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
  auto light = std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB(18));

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

int main(int argc, char const *argv[]) {
  dym::MLSMPM<dym::HighGrid, dym::OneSeparateOtherSticky> sim;
  sim.globalForce = dym::Vector3({0.f, -9.8 * 8.f, 0.f});
  std::default_random_engine re;
  std::uniform_real_distribution<Real> u(-1.f, 1.f);
  u_int n3 = 10000;
  dym::Tensor<dym::Vector3> newX(0, dym::gi(n3));

  newX.for_each_i([&](dym::Vector3 &pos) {
    pos = dym::Vector3({u(re), u(re), u(re)}) * 0.1f;
  });

  sim.addParticle(newX + dym::Vector3({0.35, 0.33, 0.5}),
                  sim.addLiquidMaterial());
  sim.addParticle(newX + dym::Vector3({0.50, 0.55, 0.5}),
                  sim.addJellyMaterial());
  sim.addParticle(newX + dym::Vector3({0.65, 0.77, 0.5}),
                  sim.addPlasticMaterial());

  const Real dt = 5e-5;
  const int volume_n = 80;

  dym::Tensor<Real> volume0(0, dym::gi(volume_n, volume_n, volume_n));
  dym::Tensor<Real> volume1(0, dym::gi(volume_n, volume_n, volume_n));
  dym::Tensor<Real> volume2(0, dym::gi(volume_n, volume_n, volume_n));
  auto Tp = [&](dym::Tensor<dym::Vector3> &x) {
    volume0 = 0;
    volume1 = 0;
    volume2 = 0;
    x.for_each_i([&](dym::Vector3 &pos, int i) {
      auto pos_off = pos * volume_n;
      auto pos_i = pos_off.cast<int>();
      if (i / n3 == 0)
        volume0[pos_i] += 0.5;
      if (i / n3 == 1)
        volume1[pos_i] += 0.5;
      if (i / n3 == 2)
        volume2[pos_i] += 0.5;
      dym::Loop<int, 2>([&](auto ii) {
        dym::Loop<int, 2>([&](auto jj) {
          dym::Loop<int, 2>([&](auto kk) {
            auto pos_ijk = pos_i + dym::Vector3i({ii, jj, kk});
            if (pos_ijk >= 0; pos_ijk < volume_n) {
              if (i / n3 == 0)
                volume0[pos_ijk] += 0.5;
              if (i / n3 == 1)
                volume1[pos_ijk] += 0.5;
              if (i / n3 == 2)
                volume2[pos_ijk] += 0.5;
              //   volume[pos_ijk] += 0.5;
            }
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
  dym::rt::RtRender render(image_width, image_height);

  // static world
  dym::rt::HittableList world;
  dym::rt::HittableList lights;

  Real begin = 0.35, end = 0.65;

  world.add(std::make_shared<dym::rt::BvhNode>(cornell_box()));
  lights.add(std::make_shared<dym::rt::xz_rect<true>>(
      begin, end, begin, end, 0.998, std::shared_ptr<dym::rt::Material>()));

  // Camera
  dym::rt::Point3 lookfrom({0.5, 0.5, -1.35});
  dym::rt::Point3 lookat({0.5, 0.5, 0});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.0;

  // dym::rt::Camera<false> cam(lookfrom, lookat, vup, 40, aspect_ratio,
  // aperture,
  //                            dist_to_focus);
  render.cam.setCamera(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
                       dist_to_focus);

  qprint(render.cam.getViewMatrix4_transform());

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  Real t = 1, t_inv = 1 - t;
  dym::TimeLog time;
  int ccc = 1;
  const int steps = 30;
  dym::Matrix3 scalem = dym::matrix::identity<Real, 3>(1.03);

  // model
  auto wmesh0 = std::make_shared<dym::rt::Mesh>();
  auto wmesh1 = std::make_shared<dym::rt::Mesh>();
  auto wmesh2 = std::make_shared<dym::rt::Mesh>();

  time.reStart();
  gui.update([&]() {
    // lookfrom[0] = 0.5 + dym::sin((Real)ccc / 10.0) * 0.2;
    // qprint(lookfrom);
    // render.cam.setCamera(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
    //                      dist_to_focus);
    dym::TimeLog partTime;
    Tp(sim.getPos());
    for (int i = 0; i < steps; ++i)
      sim.advance(dt);
    qprint("fin sim part time:", partTime.getRecord());
    partTime.reStart();
    auto mesh0 = dym::marchingCubes(volume0, 0.5);
    auto mesh1 = dym::marchingCubes(volume1, 0.5);
    auto mesh2 = dym::marchingCubes(volume2, 0.5);
    qprint("fin mc part time:", partTime.getRecord());
    partTime.reStart();

    wmesh0->reBuild(mesh0, whiteWaterSur());
    wmesh1->reBuild(mesh1, whiteJullySur());
    wmesh2->reBuild(mesh2, whiteSur());

    dym::rt::HittableList worlds;
    worlds.add(std::make_shared<dym::rt::BvhNode>(world));
    worlds.add(std::make_shared<dym::rt::Transform3>(
        wmesh0, scalem, dym::Vector3({0.5, 0.5 - 0.02, 0.5})));
    worlds.add(std::make_shared<dym::rt::Transform3>(
        wmesh1, scalem, dym::Vector3({0.5, 0.5 - 0.02, 0.5})));
    worlds.add(std::make_shared<dym::rt::Transform3>(
        wmesh2, scalem, dym::Vector3({0.5, 0.5 - 0.02, 0.5})));
    qprint("fin build worlds part time:", partTime.getRecord());
    partTime.reStart();

    render.worlds = worlds;
    render.lights = lights;

    render.render(samples_per_pixel, max_depth);

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
                 "./rt_out/sample/3/frame_" + std::to_string(ccc - 1) + ".jpg");

    // image = dym::filter2D(image, dym::Matrix3(1.f / 9.f));

    gui.imshow(image);
  });

  return 0;
}

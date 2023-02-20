#include "fun_pkg.hpp"

int main(int argc, char const *argv[]) {
  const auto aspect_ratio = 5.f / 3.f;
  const int image_width = 2000;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  int samples_per_pixel = 1;
  const int max_depth = 20;
  dym::rt::svgf_op_color_alpha = 0.05;
  dym::rt::svgf_op_moment_alpha = 0.05;
  dym::rt::svgf_op_sepcolor = true;
  dym::rt::svgf_op_addcolor = true;
  dym::Tensor<dym::Vector<Real, dym::PIC_RGB>> image(
      0, dym::gi(image_height, image_width));
  dym::Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> imageP(
      0, dym::gi(image_height, image_width));

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  Real t = 0.4, t_inv = 1 - t;
  dym::TimeLog time;
  int ccc = 1;

  // World
  dym::rt::HittableList world;
  dym::rt::HittableList lights;
  //   Real begin = 0.35, end = 0.65;
  //   lights.add(std::make_shared<dym::rt::xz_rect>(
  //       begin, end, begin, end, 0.998,
  //       std::shared_ptr<dym::rt::Material>()));

  std::vector<std::string> paths{
      "./image/magic/arrorBody.png",    "./image/magic/arrorDot.png",
      "./image/magic/arrorPattern.png", "./image/magic/cirLevel1.png",
      "./image/magic/cirLevel2.png",    "./image/magic/patternIn.png",
      "./image/magic/patternOut.png",   "./image/magic/prayer.png"};
  // std::vector<std::string> paths{"./image/magic/arrorBody.png"};
  std::vector<int> le{1, 1, 1, 3, 3, 5, 4, 2};
  dym::rt::HittableList magic;
  for (int i = 0; i < paths.size(); ++i) {
    auto &p = paths[i];
    Real hei = 0.09 - le[i] * 0.03;
    // auto mat = std::make_shared<dym::rt::magicLight>(p, 1.);
    // auto mat = magicSur(dym::rt::ColorRGB{235, 172, 0} / 255.,
    //                     dym::rt::ColorRGB{66, 0, 15} / 255. / 50., 0.2);
    auto mat = magicSur(dym::rt::ColorRGB{235, 172, 0} / 255.,
                        dym::rt::ColorRGB(0.), 0.2);
    auto obj = std::make_shared<dym::rt::xz_rect<true>>(-1, 1, -1, 1, hei, mat);
    auto mobj = std::make_shared<dym::rt::Mask>(p, obj);
    magic.add(mobj);
    // lights.add(mobj);
  }

  dym::Quaternion magGloR = dym::getQuaternion<Real>(dym::Pi / 6., {0, 1, 0}) *
                            dym::getQuaternion<Real>(dym::Pi / 12., {0, 0, 1});
  dym::Quaternion rotate =
      magGloR * dym::getQuaternion<Real>(-dym::Pi / 2., {0, 0, 1});
  dym::Matrix3 scalem = dym::matrix::identity<Real, 3>(0.1);
  dym::Vector3 translation({0.5, 0.091, 0.5});

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::BvhNode>(magic), scalem * rotate.to_matrix(),
      translation));

  //   // bunny
  //   dym::rdt::Model bunny("./assets/bunny.ply");
  //   world.addObject<dym::rt::Transform3>(
  //       std::make_shared<dym::rt::Mesh>(bunny.meshes[0], metalSur(0.8, 0.2)),
  //       dym::matrix::identity<Real, 3>(0.5) *
  //           dym::getQuaternion<Real>(dym::Pi, {0, 1, 0}).to_matrix(),
  //       dym::Vector3{0.5, 0, 0.5});

  // ring
  dym::rdt::Model loader("./assets/singel-chain-ring/ring.obj");

  rotate = magGloR * dym::getQuaternion<Real>(dym::Pi / 6., {1, 0, 0});
  scalem = dym::matrix::identity<Real, 3>(0.15);

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::Mesh>(loader.meshes[0], glassSur(1., 1.5)),
      scalem * rotate.to_matrix(), translation));

  rotate = magGloR * dym::getQuaternion<Real>(-dym::Pi / 6., {1, 0, 0});
  scalem = dym::matrix::identity<Real, 3>(0.15);

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::Mesh>(loader.meshes[0], glassSur(1., 1.5)),
      scalem * rotate.to_matrix(), translation));

  rotate = magGloR * dym::getQuaternion<Real>(dym::Pi * 11 / 12., {0, 0, 1});
  scalem = dym::matrix::identity<Real, 3>(0.1);

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::Mesh>(loader.meshes[0], glassSur(1., 1.5)),
      scalem * rotate.to_matrix(), translation));

  rotate = magGloR * dym::getQuaternion<Real>(-dym::Pi * 11 / 12., {0, 0, 1});
  scalem = dym::matrix::identity<Real, 3>(0.1);

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::Mesh>(loader.meshes[0], glassSur(1., 1.5)),
      scalem * rotate.to_matrix(), translation));

  for (int i = 0; i < 10; ++i) {
    scalem = dym::matrix::identity<Real, 3>(0.02 * i);
    translation = {0.5, 0.002 * i, 0.5};

    world.add(std::make_shared<dym::rt::Transform3>(
        std::make_shared<dym::rt::Mesh>(loader.meshes[0], glassSur(1., 1.5)),
        scalem, translation));
  }

  // connell box
  auto cb3 = cornell_box3();
  world.add(std::make_shared<dym::rt::BvhNode>(std::get<0>(cb3)));
  lights = std::get<1>(cb3);

  auto lightBall = std::make_shared<dym::rt::Sphere>(
      dym::Vector3{.9, .8, .8}, 0.1,
      std::make_shared<dym::rt::DiffuseLight>(10.));
  world.add(lightBall);
  lights.add(lightBall);
  // world.add(std::make_shared<dym::rt::BvhNode>(cornell_box()));

  // Camera
  dym::rt::Point3 lookfrom({0.6, 0.1, 0.25});
  //   dym::rt::Point3 lookfrom({0.5, 0.1, 0.25});
  dym::rt::Point3 lookat({0.5, 0.097, 0.5});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = .0;

  dym::rt::RtRender render(image_width, image_height);

  render.cam.setCamera(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
                       dist_to_focus);

  render.worlds.addObject<dym::rt::BvhNode>(world);
  render.lights = lights;

  render.registRenderKernel<dym::rt::MIS_RR_PT>();

  time.reStart();
  gui.update([&]() {
    dym::TimeLog partTime;
    render.render(
        samples_per_pixel, 0.85,
        [](const dym::rt::Ray &r) {
          auto nd = -dym::Vector3{-.5, -1, 1}.normalize();
          return dym::lerp(dym::Vector3{13, 15, 15} / 255.,
                           dym::Vector3{98, 141, 163} / 255.,
                           dym::pow(dym::max(nd.dot(r.direction()), 0.), 2));
        },
        1.);
    // samples_per_pixel = 50;

    qprint("fin render part time:", partTime.getRecord());
    partTime.reStart();

    render.denoise();

    qprint("fin denoise part time:", partTime.getRecord());
    // partTime.reStart();

    ccc++;
    time.record();
    time.reStart();
    // auto image = render.getFrameGBuffer("depth", 100);
    auto image = render.getFrame(255.);
    dym::imwrite(image,
                 "./rt_out/magicCir/frame_" + std::to_string(ccc - 1) + ".png");
    gui.imshow(image);
  });
  return 0;
}
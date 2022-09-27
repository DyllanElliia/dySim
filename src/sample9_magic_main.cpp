#include "dyMath.hpp"
#include "fun_pkg.hpp"
#include "render/object/box.hpp"
#include "tools/sugar.hpp"

int main(int argc, char const *argv[]) {
  const auto aspect_ratio = 1.f / 1.f;
  const int image_width = 800;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  int samples_per_pixel = 10;
  const int max_depth = 50;
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
  for (int i = 0; i < paths.size(); ++i) {
    auto &p = paths[i];
    Real hei = 0.05 - le[i] * 0.0025;
    // auto mat = std::make_shared<dym::rt::magicLight>(p, 1.);
    auto mat = metalSur({0.8, 0.6, 0.1}, 0.1);
    auto obj = std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, hei, mat);
    auto mobj = std::make_shared<dym::rt::Mask>(p, obj);
    world.add(mobj);
    // lights.add(mobj);
  }
  //   Real begin = 0.15, end = 0.85;
  //   lights.add(std::make_shared<dym::rt::xz_rect>(
  //       begin, end, begin, end, 0.04, std::shared_ptr<dym::rt::Material>()));
  auto cb3 = cornell_box3();
  world.add(std::make_shared<dym::rt::BvhNode>(std::get<0>(cb3)));
  lights = std::get<1>(cb3);
  // world.add(std::make_shared<dym::rt::BvhNode>(cornell_box()));
  qprint("h");
  dym::rdt::Model loader("./assets/singel-chain-ring/ring.obj");
  qprint("h");
  dym::Quaternion rotate = dym::getQuaternion<Real>(dym::Pi, {0, 1, 0});
  dym::Matrix3 scalem = dym::matrix::identity<Real, 3>(0.5);
  dym::Vector3 translation({0.5, 0.4, 0.5});

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::Mesh>(loader.meshes[0], jadeSur(0.9, 0.6)),
      scalem * rotate.to_matrix(), translation));
  qprint("h");

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

  time.reStart();
  gui.update([&]() {
    dym::TimeLog partTime;
    render.render(samples_per_pixel, max_depth);

    qprint("fin render part time:", partTime.getRecord());
    partTime.reStart();

    render.denoise();

    qprint("fin denoise part time:", partTime.getRecord());
    // partTime.reStart();

    ccc++;
    time.record();
    time.reStart();
    // auto image = render.getFrameGBuffer("depth", 100);
    auto image = render.getFrame();
    //   dym::imwrite(image,
    //                "./rt_out/sample/8/frame_" + std::to_string(ccc - 1) +
    //                ".jpg");
    gui.imshow(image);
  });
  return 0;
}
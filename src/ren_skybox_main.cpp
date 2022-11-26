#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>
#include <memory>
_DYM_FORCE_INLINE_ auto whiteMetalSur(Real fuzz = 0) {
  auto white_surface = std::make_shared<dym::rt::Metal>(
      dym::rt::ColorRGB({0.8, 1.0, 0.8}), fuzz);

  return white_surface;
}
int main() {
  const auto aspect_ratio = 1.f / 1.f;
  const int image_width = 500;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  int samples_per_pixel = 1;
  const int max_depth = 50;

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  Real t = 0.4, t_inv = 1 - t;
  dym::TimeLog time;

  // World
  dym::rt::HittableList world;
  dym::rt::HittableList lights;

  dym::rdt::Model loader("./PLYFiles/ply/Bunny10K.ply");
  dym::Quaternion rotate = dym::getQuaternion<Real>(dym::Pi, {0, 1, 0});
  dym::Matrix3 scalem = dym::matrix::identity<Real, 3>(3.5);
  dym::Vector3 translation({0.4, 0, 0.55});

  world.add(std::make_shared<dym::rt::Transform3>(
      std::make_shared<dym::rt::Mesh>(loader.meshes[0], whiteMetalSur(0.01)),
      scalem * rotate.to_matrix(), translation));

  // Camera
  dym::rt::Point3 lookfrom({0., 0.5, -1.35});
  dym::rt::Point3 lookat({0.5, 0.5, 0});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.0;

  dym::rt::RtRender render(image_width, image_height);

  render.cam.setCamera(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
                       dist_to_focus);

  render.worlds.addObject<dym::rt::BvhNode>(world);
  render.lights = lights;

  // SkyBox Texture
  std::vector<std::string> faces{"right.jpg",  "left.jpg",  "top.jpg",
                                 "bottom.jpg", "front.jpg", "back.jpg"};
  for (auto &face : faces)
    face = "./assets/skybox/" + face;

  std::vector<std::shared_ptr<dym::rt::Material>> mat_ptrs;
  for (auto &face : faces)
    mat_ptrs.push_back(std::make_shared<dym::rt::DiffuseLight>(
        std::make_shared<dym::rt::ImageTexture<3>>(face)));

  dym::rt::Skybox skybox(mat_ptrs);

  time.reStart();
  gui.update([&]() {
    dym::TimeLog partTime;
    render.render(samples_per_pixel, max_depth, [&](const dym::rt::Ray &r) {
      // return dym::Vector3(0.8);
      return skybox.sample(r) + 0.1;
    });

    qprint("fin render part time:", partTime.getRecord());
    partTime.reStart();

    // render.denoise();

    // qprint("fin denoise part time:", partTime.getRecord());
    // partTime.reStart();

    time.record();
    time.reStart();
    // auto image = render.getFrameGBuffer("depth", 100);
    auto image = render.getFrame();
    gui.imshow(image);
  });
  return 0;
}
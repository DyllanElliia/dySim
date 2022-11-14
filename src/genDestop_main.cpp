#include "render/baseClass.hpp"
#include "render/hittableList.hpp"
#include "render/object/sphere.hpp"
#include "render/texture/solidColor.hpp"
#include "tools/sugar.hpp"
#include <dyGraphic.hpp>
#include <dyMath.hpp>
#include <dyRender.hpp>
#include <memory>

int main() {
  // NOTE: global setting:
  const auto aspect_ratio = 16.f / 9.f;
  const int image_width = 1080;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int scale_val = 1;
  const int gui_width = image_width / scale_val;
  const int gui_height = image_height / scale_val;
  int samples_per_pixel = 1;
  const int max_depth = 20;

  // NOTE: Gen World
  dym::rt::HittableList world;
  auto ground_material =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB(0.7));
  world.addObject<dym::rt::Sphere>(dym::rt::Point3{0, -1000, 0}, 1000.,
                                   ground_material);

  // NOTE: Render & Camera
  dym::rt::Point3 lookfrom({0.5, 0.5, -1.35});
  dym::rt::Point3 lookat({0.5, 0.5, 0});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.0;

  dym::rt::RtRender render(image_width, image_height);

  render.cam.setCamera(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
                       dist_to_focus);

  render.worlds.addObject<dym::rt::BvhNode>(world);

  // NOTE: GUI
  dym::GUI gui("gendp");
  gui.init(gui_width, gui_height);
  dym::TimeLog time;

  // NOTE: Run
  time.reStart();
  gui.update([&] {
    dym::TimeLog patchTime;

    render.render(samples_per_pixel, max_depth);

    qprint("fin render part time:", patchTime.getRecord());
    patchTime.reStart();
  });

  return 0;
}
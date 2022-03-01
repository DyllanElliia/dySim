/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:34:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-01 17:48:53
 * @Description:
 */
#include <dyRender.hpp>
#include <dyPicture.hpp>
#include <dyGraphic.hpp>

int main(int argc, char const* argv[]) {
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 400;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 100;
  const int max_depth = 50;
  dym::Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> image(
      0, dym::gi(image_height, image_width));

  // World

  dym::rt::HittableList world;
  world.add(
      std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0, 0, -1}), 0.5));
  world.add(
      std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0, -100.5, -1}), 100));

  // Camera
  dym::rt::Camera cam;

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  int samples = 0;
  gui.update([&]() {
    image.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB>& color_p, int i,
                         int j) {
      auto u = (i + dym::rt::random_real()) / (image_width - 1);
      auto v = (j + dym::rt::random_real()) / (image_height - 1);
      dym::rt::Ray r = cam.get_ray(u, v);
      dym::rt::ColorRGB color =
          color_p.cast<Real>() + ray_color(r, world, max_depth);
      // qprint("here", color, color_p.cast<Real>());
      color /= 2.f;
      // qprint("here/", color);
      color_p =
          dym::clamp(color * 255.f, dym::Vector3(0.f), dym::Vector3(255.99f))
              .cast<dym::Pixel>();
      // qprint(u, v, color, color_p);
      // getchar();
    });
    samples++;
    // qprint(samples);
    gui.imshow(image);
    dym::imwrite(image, "./image_out/rt_test.jpg");
    qprint("fin");
    getchar();
  });
  return 0;
}

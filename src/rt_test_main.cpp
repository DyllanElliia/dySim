/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:34:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-03 17:00:57
 * @Description:
 */
#include <dyRender.hpp>
#include <dyPicture.hpp>
#include <dyGraphic.hpp>

int main(int argc, char const* argv[]) {
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 800;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 5;
  const int max_depth = 10;
  dym::Tensor<dym::Vector<Real, dym::PIC_RGB>> image(
      0, dym::gi(image_height, image_width));
  dym::Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> imageP(
      0, dym::gi(image_height, image_width));
  // World

  dym::rt::HittableList world;

  auto material_ground =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({0.8, 0.8, 0.0}));
  auto material_center =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({0.1, 0.2, 0.5}));
  auto material_left = std::make_shared<dym::rt::Dielectric>(1.5);
  auto material_right =
      std::make_shared<dym::rt::Metal>(dym::rt::ColorRGB({0.8, 0.6, 0.2}));

  world.add(std::make_shared<dym::rt::Sphere>(
      dym::rt::Point3({0.0, -100.5, -1.0}), 100.0, material_ground));
  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0.0, 0.0, -1.0}),
                                              0.5, material_center));
  world.add(std::make_shared<dym::rt::Sphere>(
      dym::rt::Point3({-1.0, 0.0, -1.0}), 0.5, material_left));
  world.add(std::make_shared<dym::rt::Sphere>(
      dym::rt::Point3({-1.0, 0.0, -1.0}), -0.4, material_left));
  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({1.0, 0.0, -1.0}),
                                              0.5, material_right));

  // Camera
  dym::rt::Point3 lookfrom({-2, 2, 1});
  dym::rt::Point3 lookat({0, 0, -1});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.f;

  dym::rt::Camera<true> cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture,
                            dist_to_focus);

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  Real t = 0.4, t_inv = 1 - t;
  dym::TimeLog time;
  int ccc = 0;
  gui.update([&]() {
    image.for_each_i([&](dym::Vector<Real, dym::PIC_RGB>& color, int i, int j) {
      auto oldColor = color;
      color = 0.f;
      for (int samples = 0; samples < samples_per_pixel; samples++) {
        auto u = (j + dym::rt::random_real()) / (image_width - 1);
        auto v = (i + dym::rt::random_real()) / (image_height - 1);
        dym::rt::Ray r = cam.get_ray(u, v);
        color += ray_color(r, world, max_depth);
      }
      // qprint("here", color, color_p.cast<Real>());
      // color /= Real(samples_per_pixel);
      color = dym::sqrt(color * (1.f / Real(samples_per_pixel)));
      // qprint("here/", color);
      color =
          dym::clamp(color * 255.f, dym::Vector3(0.f), dym::Vector3(255.99f));
      if (ccc) color = color * t + oldColor * t_inv;
      // qprint(u, v, color, color_p);
      // getchar();
    });
    ccc++;
    time.record();
    time.reStart();
    image = dym::filter2D(image, dym::Matrix3(1.f / 9.f));
    imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB>& e, int i) {
      e = image[i].cast<dym::Pixel>();
    });
    gui.imshow(imageP);
    // dym::imwrite(image, "./image_out/rt_test.jpg");
    // qprint("fin");
    // getchar();
  });
  return 0;
}

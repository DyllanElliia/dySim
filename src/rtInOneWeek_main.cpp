/*
 * @Author: DyllanElliia
 * @Date: 2022-03-04 13:50:22
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-05-23 15:25:38
 * @Description:
 */
/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:34:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-04 13:14:56
 * @Description:
 */
#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>

dym::rt::BvhNode random_scene() {
  dym::rt::HittableList world;

  auto ground_material =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({0.5, 0.5, 0.5}));
  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0, -1000, 0}),
                                              1000, ground_material));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = dym::rt::random_real();
      dym::rt::Point3 center({a + 0.9f * dym::rt::random_real(), 0.2f,
                              b + 0.9f * dym::rt::random_real()});

      if ((center - dym::rt::Point3({4, 0.2, 0})).length() > 0.9) {
        std::shared_ptr<dym::rt::Material> sphere_material;

        if (choose_mat < 0.5) {
          // diffuse
          auto albedo = dym::rt::vec_random() * dym::rt::vec_random();
          sphere_material = std::make_shared<dym::rt::Lambertian>(albedo);
          world.add(
              std::make_shared<dym::rt::Sphere>(center, 0.2, sphere_material));
        } else if (choose_mat < 0.8) {
          // metal
          auto albedo = dym::rt::vec_random(0.5, 1);
          auto fuzz = dym::rt::random_real(0, 0.5);
          sphere_material = std::make_shared<dym::rt::Metal>(albedo, fuzz);
          world.add(
              std::make_shared<dym::rt::Sphere>(center, 0.2, sphere_material));
        } else {
          // glass
          sphere_material = std::make_shared<dym::rt::Dielectric>(1.5);
          world.add(
              std::make_shared<dym::rt::Sphere>(center, 0.2, sphere_material));
        }
      }
    }
  }

  auto material1 = std::make_shared<dym::rt::Dielectric>(1.5);
  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({0, 1, 0}), 1.0,
                                              material1));

  auto material2 =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({0.4, 0.2, 0.1}));
  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({-4, 1, 0}), 1.0,
                                              material2));

  auto material3 =
      std::make_shared<dym::rt::Metal>(dym::rt::ColorRGB({0.7, 0.6, 0.5}), 0.0);
  world.add(std::make_shared<dym::rt::Sphere>(dym::rt::Point3({4, 1, 0}), 1.0,
                                              material3));

  return dym::rt::BvhNode(world);
}

int main(int argc, char const *argv[]) {
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 1800;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 1;
  const int max_depth = 50;
  dym::Tensor<dym::Vector<Real, dym::PIC_RGB>> image(
      0, dym::gi(image_width, image_height));
  dym::Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> imageP(
      0, dym::gi(image_width, image_height));
  // World

  auto world = random_scene();

  // Camera
  dym::rt::Point3 lookfrom({13, 2, 3});
  dym::rt::Point3 lookat({0, 0, 0});
  dym::Vector3 vup({0, 1, 0});
  auto aperture = 0.5f;
  auto dist_to_focus = 10.f;

  dym::rt::Camera<true> cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture,
                            dist_to_focus);

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  Real t = 0.4, t_inv = 1 - t;
  dym::TimeLog time;
  int ccc = 1;
  gui.update([&]() {
    image.for_each_i([&](dym::Vector<Real, dym::PIC_RGB> &color, int i, int j) {
      auto oldColor = color / 255.f * Real((ccc - 1) * samples_per_pixel);

      color = 0.f;
      for (int samples = 0; samples < samples_per_pixel; samples++) {
        auto u = (i + dym::rt::random_real()) / (image_width - 1);
        auto v = (j + dym::rt::random_real()) / (image_height - 1);
        dym::rt::Ray r = cam.get_ray(u, v);
        color += ray_color_pdf(
            r, world, nullptr, max_depth, [](const dym::rt::Ray &r) {
              dym::Vector3 unit_direction = r.direction().normalize();
              Real t = 0.5f * (unit_direction.y() + 1.f);
              return (1.f - t) * dym::rt::ColorRGB(1.f) +
                     t * dym::rt::ColorRGB({0.5f, 0.7f, 1.0f});
            });
      }
      // qprint("here", color, color_p.cast<Real>());
      // color /= Real(samples_per_pixel);
      color = dym::sqrt(color * (1.f / Real(samples_per_pixel)));
      // qprint("here/", color);
      color = (color + oldColor) / Real(samples_per_pixel * ccc);
      color =
          dym::clamp(color * 255.f, dym::Vector3(0.f), dym::Vector3(255.99f));

      // qprint(u, v, color, color_p);
      // getchar();
    });
    ccc++;
    time.record();
    time.reStart();
    // image = dym::filter2D(image, dym::Matrix3(1.f / 9.f));
    imageP.for_each_i([&](dym::Vector<dym::Pixel, dym::PIC_RGB> &e, int i) {
      e = image[i].cast<dym::Pixel>();
    });
    gui.imshow(imageP);
    dym::imwrite(image,
                 "./rt_out/rtOneWeek/rt_test_" + std::to_string(ccc) + ".jpg");
    // qprint("fin");
    // getchar();
  });
  return 0;
}

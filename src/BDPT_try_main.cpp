#include "BDPT_try.hpp"
#include "dyMath.hpp"
#include "render/camera.hpp"
#include "render/randomFun.hpp"
#include <cstdio>
#include <unistd.h>

int main() {
  const auto aspect_ratio = 5.f / 3.f;
  const int image_width = 2000;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  dym::rt::Point3 lookfrom({0, 0, -1});
  //   dym::rt::Point3 lookfrom({0.5, 0.1, 0.25});
  dym::rt::Point3 lookat({0, 0, 1});
  dym::Vector3 vup({0, 1, 0});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 0.0;
  dym::rt::Camera cam(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
                      dist_to_focus);

  while (true) {
    auto u = dym::rt::random_real(), v = dym::rt::random_real();
    auto r = cam.get_ray(u, v);
    // auto [we, coord, w] = cam.eval_we(r);
    // auto [la_inv, pdf] = cam.pdf_we(r);
    // qprint(w, r.dir.normalize());
    // qprint("cos:", w.dot(r.dir.normalize()));
    qprint(u, v);
    // qprint(we, dym::uv2st(coord), w);
    // // qprint("oo:",r.orig,)
    // qprint(la_inv, pdf);

    auto pos = r.orig + 1.1 * r.dir;
    auto pos2 = cam.getViewMatrix4_transform() * dym::Vector4(pos, 1);
    auto pos3 = cam.getViewMatrix4_Perspective() *
                cam.getViewMatrix4_transform() * dym::Vector4(pos, 1);
    auto pos31 = pos3 / pos3[3];
    qprint(pos, pos2, pos31, pos31 / pos31[2] * pos2[2]);
    qprint(1 - (dym::Vector2(pos31) / 2 + 0.5));
    char c = getchar();
    qprint("Input", c);
    if (c == 'e')
      break;
  }
  return 0;
}
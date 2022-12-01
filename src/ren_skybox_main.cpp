#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>
#include <memory>
_DYM_FORCE_INLINE_ auto whiteMetalSur(Real fuzz = 0) {
  auto white_surface = std::make_shared<dym::rt::Metal>(
      dym::rt::ColorRGB({0.8, 1.0, 0.8}), fuzz);

  return white_surface;
}

namespace dym {
namespace rt {
class SkyboxT : public Hittable {
public:
  SkyboxT() {}
  SkyboxT(const std::vector<shared_ptr<Material>> &mat_ptrs) {
    if (mat_ptrs.size() < 6) {
      DYM_WARNING_cs("SkyBox", "Skybox need 6 picture path for generate.");
      return;
    }
    box_min = -1e7;
    box_max = 1e7;
    auto &p0 = box_min, &p1 = box_max;
    HittableList side;

    side.add(make_shared<xy_rect<true>>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(),
                                        mat_ptrs[5]));
    side.add(make_shared<xy_rect<false>>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(),
                                         mat_ptrs[4]));

    side.add(make_shared<xz_rect<false>>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(),
                                         mat_ptrs[2]));
    side.add(make_shared<xz_rect<false>>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(),
                                         mat_ptrs[3], true));

    side.add(make_shared<yz_rect<false>>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(),
                                         mat_ptrs[1]));
    side.add(make_shared<yz_rect<true>>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(),
                                        mat_ptrs[0]));

    sides = BvhNode(side);
  }
  ~SkyboxT() {}

  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const override {
    Ray ro = r;
    ro.orig = 0.0;
    return sides.hit(ro, t_min, t_max, rec);
  }
  virtual bool bounding_box(aabb &output_box) const override {
    output_box = aabb(box_min, box_max);
    return true;
  }

  _DYM_FORCE_INLINE_ ColorRGB sample(const Ray &r) {
    HitRecord rec;
    // if (!hit(ro, 1e-7, infinity, rec))
    //   return 0.0;
    hit(r, 1e-7, infinity, rec);
    auto &hitMat = *(rec.mat_ptr);
    ColorRGB Le = rec.mat_ptr->emitted(r, rec);

    return Le;
  }

private:
  // shared_ptr<Material> mat_ptrs[6];
  Point3 box_min;
  Point3 box_max;
  BvhNode sides;
};

} // namespace rt
} // namespace dym
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

  render.cam.setCamera(lookfrom, lookat, vup, 80, aspect_ratio, aperture,
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

  dym::rt::SkyboxT skybox(mat_ptrs);

  int ccc = 0;
  time.reStart();
  gui.update([&]() {
    lookfrom[0] = 0.5;
    lookfrom[1] = 3 * dym ::sin(Real(ccc) / 10.);
    lookfrom[2] = 3 * dym ::cos(Real(ccc) / 10.);
    vup = {1, 0, 0};
    ccc++;
    render.cam.setCamera(lookfrom, lookat, vup, 80, aspect_ratio, aperture,
                         dist_to_focus);
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
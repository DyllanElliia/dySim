#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>
#include <memory>

namespace dym {
namespace rt {
class fuzzDielectric : public Material {
public:
  fuzzDielectric(const Real &index_of_refraction, const Real fuzz = -1.f)
      : ir(index_of_refraction), albedo(make_shared<SolidColor>(ColorRGB(1.f))),
        fuzz(fuzz) {}
  fuzzDielectric(const ColorRGB &color, const Real &index_of_refraction,
                 const Real fuzz = -1.f)
      : ir(index_of_refraction), albedo(make_shared<SolidColor>(color)),
        fuzz(fuzz) {}
  fuzzDielectric(const shared_ptr<Texture> &tex,
                 const Real &index_of_refraction, const Real fuzz = -1.f)
      : ir(index_of_refraction), albedo(tex), fuzz(fuzz) {}

  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const override {
    srec.is_specular = true;
    srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
    Real refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

    Vector3 unit_direction = r_in.direction().normalize();
    Real cos_theta = fmin(dym::vector::dot(-unit_direction, rec.normal), 1.f);
    Real sin_theta = dym::sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.f;
    Vector3 direction;

    if (cannot_refract ||
        reflectance(cos_theta, refraction_ratio) > random_real())
      direction = unit_direction.reflect(rec.normal);
    else
      direction = refract(unit_direction, rec.normal, refraction_ratio);

    if (fuzz > 0)
      direction = (direction + fuzz * random_in_unit_sphere()).normalize();
    srec.specular_ray = Ray(rec.p, direction, r_in.time());
    srec.pdf_ptr = nullptr;
    ;
    return true;
  }

  virtual Vector3 BxDF_Evaluate(const Ray &r_in, const Ray &scattered,
                                const HitRecord &rec,
                                const ScatterRecord &srec) const {
    if (rec.front_face)
      return ColorRGB(1.);
    // auto len = 1 - dym::pow(dym::exp(-rec.t), 3);
    auto len = 1. / (1 + dym::exp(1 - 25 * rec.t));
    auto albedo_len = dym::exp(-rec.t);

    // if (random_real() < 1e-6)
    //   qprint(r_in.dir.length(), (rec.p - r_in.origin()) / r_in.dir, rec.t,
    //          r_in.origin(), rec.p);

    return dym::lerp(ColorRGB(1.f), srec.attenuation * albedo_len, len);
    // return {dym::min(len, 1), 0.1, 0.1};
  }

public:
  Real ir; // Index of Refraction
  shared_ptr<Texture> albedo;
  Real fuzz;

private:
  _DYM_FORCE_INLINE_ Vector3 refract(const Vector3 &uv, const Vector3 &n,
                                     const Real &etai_over_etat) const {
    auto cos_theta = fmin(dym::vector::dot(-uv, n), 1.f);
    Vector3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vector3 r_out_parallel =
        Real(-dym::sqrt(dym::abs(1.0 - r_out_perp.length_sqr()))) * n;
    return r_out_perp + r_out_parallel;
  }
  static _DYM_FORCE_INLINE_ Real reflectance(const Real &cosine,
                                             const Real &ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
  }
};

class Transformtest : public Hittable {
public:
  Transformtest(const shared_ptr<Hittable> &ptr, const Matrix3 &mat,
                const Vector3 &offset = 0)
      : ptr(ptr), mat(mat), offset(offset) {
    mat_inv = mat.inverse();
    mat_norm_it = mat.inverse().transpose();
    hasbox = ptr->bounding_box(bbox);

    const auto infinity = std::numeric_limits<Real>::infinity();
    Point3 minp(infinity), maxp(-infinity);
    const auto &bmin = bbox.min(), &bmax = bbox.max();

    Loop<int, 2>([&](auto i) {
      Loop<int, 2>([&](auto j) {
        Loop<int, 2>([&](auto k) {
          Point3 tester({i * bmax.x() + (1 - i) * bmin.x(),
                         j * bmax.y() + (1 - j) * bmin.y(),
                         k * bmax.z() + (1 - k) * bmin.z()});
          tester = mat * tester;
          minp = min(minp, tester), maxp = max(maxp, tester);
        });
      });
    });
    bbox = aabb(minp + offset, maxp + offset);

    Real resm = 0;
    Matrix3 mmm = mat;
    mmm.for_each([&](Real &e, int i, int j) { resm += e * e; });
    qprint(resm);
  }

  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const override;
  virtual bool bounding_box(aabb &output_box) const override;

public:
  shared_ptr<Hittable> ptr;
  Matrix3 mat, mat_inv, mat_norm_it;
  Vector3 offset;
  bool hasbox;
  aabb bbox;
};

bool Transformtest::hit(const Ray &r, Real t_min, Real t_max,
                        HitRecord &rec) const {
  auto origin = mat_inv * (r.origin() - offset);
  auto direction = (mat_inv * r.direction()).normalize();
  Ray tf_r(origin, direction, r.time());

  if (!ptr->hit(tf_r, 1e-6, infinity, rec))
    return false;

  auto oldp = rec.p;
  rec.p = mat * rec.p + offset;
  rec.normal = (mat_norm_it * rec.normal).normalize();

  auto oldt = rec.t;
  rec.t = (rec.p - r.origin())[0] / r.direction()[0];

  return (rec.t > t_min && rec.t < t_max);
}
bool Transformtest::bounding_box(aabb &output_box) const {
  output_box = bbox;
  return hasbox;
}
} // namespace rt
} // namespace dym

int main() {
  // NOTE: global setting:
  const auto aspect_ratio = 16.f / 9.f;
  const int image_width = 1080;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int scale_val = 1;
  const int gui_width = image_width / scale_val;
  const int gui_height = image_height / scale_val;
  int samples_per_pixel = 3;
  const int max_depth = 20;

  // NOTE: GUI
  dym::GUI gui("gendp");
  gui.init(gui_width, gui_height);
  dym::TimeLog time;

  // NOTE: Gen World
  dym::rt::HittableList world;
  auto ground_material =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB(0.7));
  auto fuzzDie_material = std::make_shared<dym::rt::fuzzDielectric>(
      dym::rt::ColorRGB({0.1, 0.3, 0.1}), 1.5, 0.1);
  auto fuzzDieBunny_material = std::make_shared<dym::rt::fuzzDielectric>(
      dym::rt::ColorRGB({0.1, 0.3, 0.1}), 1.5, 0.15);
  world.addObject<dym::rt::Sphere>(dym::rt::Point3{0, -1000, 0}, 1000.,
                                   ground_material);

  dym::rdt::Model loader("./PLYFiles/ply/Bunny10K.ply");
  dym::Quaternion rotate = dym::getQuaternion<Real>(dym::Pi, {0, 1, 0});
  dym::Matrix3 scalem = dym::matrix::identity<Real, 3>(3.5);
  dym::Vector3 translation({0.4, 0, 0.55});

  auto bunnyPtr =
      std::make_shared<dym::rt::Mesh>(loader.meshes[0], fuzzDieBunny_material);
  for (auto &v : bunnyPtr->vertices)
    v.normal = -v.normal;
  world.add(std::make_shared<dym::rt::Transformtest>(
      bunnyPtr, scalem * rotate.to_matrix(), translation));

  world.addObject<dym::rt::Sphere>(dym::rt::Point3{-.1, 0.3, 0.5}, 0.3,
                                   fuzzDie_material);

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

  // NOTE: Run
  time.reStart();
  gui.update([&] {
    dym::TimeLog patchTime;

    render.render(samples_per_pixel, max_depth, [](const dym::rt::Ray &r) {
      dym::Vector3 unit_direction = r.direction().normalize();
      Real t = 0.5f * (unit_direction.y() + 1.f);
      return (1.f - t) * dym::rt::ColorRGB(1.f) +
             t * dym::rt::ColorRGB({0.5f, 0.7f, 1.0f});
    });

    qprint("fin render part time:", patchTime.getRecord());
    patchTime.reStart();
    render.denoise();
    auto image = render.getFrame();
    // auto image = render.getFrameGBuffer("normal");
    // dym::imwrite(image,
    //              "./rt_out/desktop/frame_" + std::to_string(ccc - 1) +
    //              ".jpg");
    gui.imshow(image);
  });

  return 0;
}
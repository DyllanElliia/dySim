/*
 * @Author: DyllanElliia
 * @Date: 2022-03-15 15:31:02
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-18 18:16:52
 * @Description:
 */

#pragma once
#include "../../baseClass.hpp"
#include "../../material/isotropic.hpp"

namespace dym {
namespace rt {
class ConstantMedium : public Hittable {
public:
  ConstantMedium(shared_ptr<Hittable> b, Real d, shared_ptr<Texture> a)
      : boundary(b), neg_inv_density(-1 / d),
        phase_function(make_shared<isotropic>(a)) {}

  ConstantMedium(shared_ptr<Hittable> b, double d, ColorRGB c)
      : boundary(b), neg_inv_density(-1 / d),
        phase_function(make_shared<isotropic>(c)) {}

  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const override;

  virtual bool bounding_box(aabb &output_box) const override {
    return boundary->bounding_box(output_box);
  }

  virtual Real pdf_value(const Point3 &origin, const Vector3 &v) const override;
  virtual Vector3 random(const Point3 &origin) const override;

public:
  shared_ptr<Hittable> boundary;
  shared_ptr<Material> phase_function;
  Real neg_inv_density;
};

bool ConstantMedium::hit(const Ray &r, Real t_min, Real t_max,
                         HitRecord &rec) const {
  // Print occasional samples when debugging. To enable, set enableDebug true.
  const bool enableDebug = false;
  const bool debugging = enableDebug && random_real() < 1e-4;

  HitRecord rec1, rec2;

  if (!boundary->hit(r, -infinity, infinity, rec1))
    return false;
  if (debugging)
    qprint("in1", rec1.t, r.origin() + rec1.t * r.direction(), rec1.p);
  if (!boundary->hit(r, rec1.t + 1e-7, infinity, rec2))
    return false;
  // if (debugging)
  //   qprint("in2");
  if (debugging)
    qprint("t_min =", rec1.t, ", t_max =", rec2.t);

  if (rec1.t < t_min)
    rec1.t = t_min;
  if (rec2.t > t_max)
    rec2.t = t_max;

  if (rec1.t >= rec2.t)
    return false;

  if (rec1.t < 0)
    rec1.t = 0;

  const auto ray_length = r.direction().length();
  const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
  const auto hit_distance = neg_inv_density * log(random_real());

  if (hit_distance > distance_inside_boundary)
    return false;

  rec.t = rec1.t + hit_distance / ray_length;
  rec.p = r.at(rec.t);

  if (debugging) {
    qprint("hit_distance = ", hit_distance);
    qprint("rec.t = ", rec.t);
    qprint("rec.p = ", rec.p);
  }

  rec.normal = Vector3({1, 0, 0}); // arbitrary
  rec.front_face = true;           // also arbitrary
  rec.mat_ptr = phase_function;
  rec.obj_id = (int)(std::size_t)this;

  return true;
}

Real ConstantMedium::pdf_value(const Point3 &origin, const Vector3 &v) const {
  return 1;
}
Vector3 ConstantMedium::random(const Point3 &origin) const {
  return neg_inv_density * random_in_unit_sphere();
}
} // namespace rt
} // namespace dym
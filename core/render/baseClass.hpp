/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:58:59
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-20 14:20:46
 * @Description:
 */
#pragma once
#include "BVH/aabb.hpp"
// #include "pdf/pdf.hpp"
#include "ray.hpp"
#include "texture/solidColor.hpp"

namespace dym {
namespace rt {
class Hittable {
 public:
  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const = 0;
  virtual bool bounding_box(aabb& output_box) const = 0;

  virtual Real pdf_value(const Point3& o, const Vector3& v) const {
    return 0.0;
  }
  virtual Vector3 random(const Vector3& o) const { return Vector3({1, 0, 0}); }
};

class Material {
 public:
  virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                       ScatterRecord& srec) const {
    return false;
  }

  virtual Real scattering_pdf(const Ray& r_in, const HitRecord& rec,
                              const Ray& scattered) const {
    return 1;
  }
  virtual ColorRGB emitted(const Ray& r_in, const HitRecord& rec, Real u,
                           Real v, const Point3& p) const {
    return ColorRGB(0.f);
  }
};

class onb {
 public:
  onb() {}

  _DYM_FORCE_INLINE_ Vector3 operator[](const int& i) const { return axis[i]; }

  _DYM_FORCE_INLINE_ Vector3 u() const { return axis[0]; }
  _DYM_FORCE_INLINE_ Vector3 v() const { return axis[1]; }
  _DYM_FORCE_INLINE_ Vector3 w() const { return axis[2]; }

  _DYM_FORCE_INLINE_ Vector3 local(const Real& a, const Real& b,
                                   const Real& c) const {
    return a * u() + b * v() + c * w();
  }

  _DYM_FORCE_INLINE_ Vector3 local(const Vector3& a) const {
    return a.x() * u() + a.y() * v() + a.z() * w();
  }

  _DYM_FORCE_INLINE_ void build_from_w(const Vector3& n) {
    axis[2] = n.normalize();
    Vector3 a = (fabs(w().x()) > 0.9) ? Vector3({0, 1, 0}) : Vector3({1, 0, 0});
    axis[1] = (w().cross(a)).normalize();
    axis[0] = w().cross(v());
  }

 public:
  Vector3 axis[3];
};

class GBuffer {
 public:
  Vector3 normal;
  Vector3 position;
  Vector3 albedo;
  int obj_id;
  GBuffer(int asdf = 0) : obj_id(-1) {}
  friend std::ostream& operator<<(std::ostream& output, GBuffer& gbuffer) {
    output << "normal: " << gbuffer.normal << std::endl;
    output << "position: " << gbuffer.position << std::endl;
    output << "albedo: " << gbuffer.albedo << std::endl;
    output << "obj_id: " << gbuffer.obj_id << std::endl;
    return output;
  }
};
}  // namespace rt
}  // namespace dym
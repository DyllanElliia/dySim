/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 14:58:59
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-06-20 17:46:46
 * @Description:
 */
#pragma once
// #include "BVH/aabb.hpp"
// #include "pdf/pdf.hpp"
#include "dyMath.hpp"
#include "math/define.hpp"
#include "ray.hpp"
#include "render/randomFun.hpp"
#include "texture/solidColor.hpp"

namespace dym {
#ifndef _dym_pic_rgb_
#define _dym_pic_rgb_
const std::size_t PIC_GRAY = 1;
const std::size_t PIC_RGB = 3;
#endif
namespace rt {

class onb {
public:
  onb() {}

  _DYM_FORCE_INLINE_ Vector3 operator[](const int &i) const { return axis[i]; }

  _DYM_FORCE_INLINE_ Vector3 u() const { return axis[0]; }
  _DYM_FORCE_INLINE_ Vector3 v() const { return axis[1]; }
  _DYM_FORCE_INLINE_ Vector3 w() const { return axis[2]; }

  _DYM_FORCE_INLINE_ Vector3 local(const Real &a, const Real &b,
                                   const Real &c) const {
    return a * u() + b * v() + c * w();
  }

  _DYM_FORCE_INLINE_ Vector3 local(const Vector3 &a) const {
    return a.x() * u() + a.y() * v() + a.z() * w();
  }

  _DYM_FORCE_INLINE_ void build_from_w(const Vector3 &n) {
    axis[2] = n.normalize();
    Vector3 a = (fabs(w().x()) > 0.9) ? Vector3({0, 1, 0}) : Vector3({1, 0, 0});
    axis[1] = (w().cross(a)).normalize();
    axis[0] = w().cross(v());
  }

public:
  Vector3 axis[3];
};

class aabb;

class Hittable {
public:
  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const = 0;
  virtual bool bounding_box(aabb &output_box) const = 0;

  virtual Real pdf_value(const Point3 &o, const Vector3 &v) const {
    return 0.0;
  }
  virtual Vector3 random(const Vector3 &o) const { return Vector3({1, 0, 0}); }
  virtual Photon random_photon() const { return Photon(); }
};

class Material {
public:
  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const {
    return false;
  }

  virtual Vector3 BxDF_Evaluate(const Ray &r_in, const Ray &scattered,
                                const HitRecord &rec,
                                const ScatterRecord &srec) const {
    return 1.;
  }

  virtual ColorRGB emitted(const Ray &r_in, const HitRecord &rec) const {
    return ColorRGB(0.f);
  }

  virtual Ray gen_photon_r(const Point3 &p, const Vector3 &n) {
    return Ray(p, n);
  }
};

class GBuffer {
public:
  Vector3 normal;
  Vector3 position;
  Vector3 albedo;
  int obj_id;
  GBuffer(int asdf = 0) : obj_id(-1) {}
  friend std::ostream &operator<<(std::ostream &output, GBuffer &gbuffer) {
    output << "normal: " << gbuffer.normal << std::endl;
    output << "position: " << gbuffer.position << std::endl;
    output << "albedo: " << gbuffer.albedo << std::endl;
    output << "obj_id: " << gbuffer.obj_id << std::endl;
    return output;
  }
};

_DYM_FORCE_INLINE_ Photon gen_photon(shared_ptr<Material> mp,
                                     const Point3 &random_point,
                                     const Vector3 &normal) {
  return Photon(mp->gen_photon_r(random_point, normal), mp);
}
} // namespace rt
} // namespace dym

#include "BVH/aabb.hpp"
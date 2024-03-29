/*
 * @Author: DyllanElliia
 * @Date: 2022-03-07 16:43:00
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-19 13:43:45
 * @Description:
 */
#pragma once

#include "../baseClass.hpp"
#include "render/ray.hpp"

namespace dym {
namespace rt {
template <bool frontFace = true> class xy_rect : public Hittable {
public:
  xy_rect() {}

  xy_rect(Real _x0, Real _x1, Real _y0, Real _y1, Real _k,
          shared_ptr<Material> mat, bool inv_v = false)
      : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat), inv_v(inv_v){};

  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const override;

  virtual bool bounding_box(aabb &output_box) const override {
    // The bounding box must have non-zero width in each dimension, so pad the Z
    // dimension a small amount.
    output_box = aabb(Point3({x0, y0, k - 1e-4f}), Point3({x1, y1, k + 1e-4f}));
    return true;
  }

  virtual Real pdf_value(const Point3 &origin, const Vector3 &v) const override;
  virtual Vector3 random(const Point3 &origin) const override;
  virtual Photon random_photon() const override;

public:
  shared_ptr<Material> mp;
  Real x0, x1, y0, y1, k;
  bool inv_v = false;
};

template <bool frontFace = true> class yz_rect : public Hittable {
public:
  yz_rect() {}

  yz_rect(Real _y0, Real _y1, Real _z0, Real _z1, Real _k,
          shared_ptr<Material> mat, bool inv_v = false)
      : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat), inv_v(inv_v){};

  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const override;

  virtual bool bounding_box(aabb &output_box) const override {
    // The bounding box must have non-zero width in each dimension, so pad the Z
    // dimension a small amount.
    output_box = aabb(Point3({k - 1e-4f, y0, z0}), Point3({k + 1e-4f, y1, z1}));
    return true;
  }

  virtual Real pdf_value(const Point3 &origin, const Vector3 &v) const override;
  virtual Vector3 random(const Point3 &origin) const override;
  virtual Photon random_photon() const override;

public:
  shared_ptr<Material> mp;
  Real y0, y1, z0, z1, k;
  bool inv_v = false;
};

template <bool frontFace = true> class xz_rect : public Hittable {
public:
  xz_rect() {}

  xz_rect(Real _x0, Real _x1, Real _z0, Real _z1, Real _k,
          shared_ptr<Material> mat, bool inv_v = false)
      : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat), inv_v(inv_v){};

  virtual bool hit(const Ray &r, Real t_min, Real t_max,
                   HitRecord &rec) const override;

  virtual bool bounding_box(aabb &output_box) const override {
    // The bounding box must have non-zero width in each dimension, so pad the Y
    // dimension a small amount.
    output_box = aabb(Point3({x0, k - 1e-4f, z0}), Point3({x1, k + 1e-4f, z1}));
    return true;
  }

  virtual Real pdf_value(const Point3 &origin, const Vector3 &v) const override;
  virtual Vector3 random(const Point3 &origin) const override;
  virtual Photon random_photon() const override;

public:
  shared_ptr<Material> mp;
  Real x0, x1, z0, z1, k;
  bool inv_v = false;
};

template <bool frontFace>
bool xy_rect<frontFace>::hit(const Ray &r, Real t_min, Real t_max,
                             HitRecord &rec) const {
  auto t = (k - r.origin().z()) / r.direction().z();
  if (t < t_min || t > t_max)
    return false;
  auto x = r.origin().x() + t * r.direction().x();
  auto y = r.origin().y() + t * r.direction().y();
  if (x < x0 || x > x1 || y < y0 || y > y1)
    return false;
  rec.u = (x - x0) / (x1 - x0);
  rec.v = (y - y0) / (y1 - y0);
  if constexpr (!frontFace)
    rec.u = 1 - rec.u;
  if (inv_v)
    rec.v = 1 - rec.v;
  rec.t = t;
  Vector3 outward_normal({0, 0, 1});
  if constexpr (!frontFace)
    outward_normal[2] = -1;
  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mp;
  rec.p = r.at(t);
  rec.obj_id = (int)(std::size_t)this;
  return true;
}

template <bool frontFace>
bool xz_rect<frontFace>::hit(const Ray &r, Real t_min, Real t_max,
                             HitRecord &rec) const {
  auto t = (k - r.origin().y()) / r.direction().y();
  if (t < t_min || t > t_max)
    return false;
  auto x = r.origin().x() + t * r.direction().x();
  auto z = r.origin().z() + t * r.direction().z();
  if (x < x0 || x > x1 || z < z0 || z > z1)
    return false;
  rec.u = (x - x0) / (x1 - x0);
  rec.v = (z - z0) / (z1 - z0);
  if constexpr (!frontFace)
    rec.u = 1 - rec.u;
  if (inv_v)
    rec.v = 1 - rec.v;

  rec.t = t;
  Vector3 outward_normal({0, 1, 0});
  if constexpr (!frontFace)
    outward_normal[1] = -1;
  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mp;
  rec.p = r.at(t);
  rec.obj_id = (int)(std::size_t)this;
  return true;
}

template <bool frontFace>
bool yz_rect<frontFace>::hit(const Ray &r, Real t_min, Real t_max,
                             HitRecord &rec) const {
  auto t = (k - r.origin().x()) / r.direction().x();
  if (t < t_min || t > t_max)
    return false;
  auto y = r.origin().y() + t * r.direction().y();
  auto z = r.origin().z() + t * r.direction().z();
  if (y < y0 || y > y1 || z < z0 || z > z1)
    return false;
  rec.u = (z - z0) / (z1 - z0);
  rec.v = (y - y0) / (y1 - y0);
  if constexpr (!frontFace)
    rec.u = 1 - rec.u;
  if (inv_v)
    rec.v = 1 - rec.v;
  rec.t = t;
  Vector3 outward_normal({1, 0, 0});
  if constexpr (!frontFace)
    outward_normal[0] = -1;
  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mp;
  rec.p = r.at(t);
  rec.obj_id = (int)(std::size_t)this;
  return true;
}

template <bool frontFace>
Real xy_rect<frontFace>::pdf_value(const Point3 &origin,
                                   const Vector3 &v) const {
  HitRecord rec;
  if (!this->hit(Ray(origin, v), 0.001, infinity, rec))
    return 0;

  auto area = (x1 - x0) * (y1 - y0);
  auto distance_squared = rec.t * rec.t * v.length_sqr();
  auto cosine = fabs(v.dot(rec.normal) / v.length());

  return distance_squared / (cosine * area);
}
template <bool frontFace>
Vector3 xy_rect<frontFace>::random(const Point3 &origin) const {
  auto random_point = Point3({random_real(x0, x1), random_real(y0, y1), k});
  return random_point - origin;
}
template <bool frontFace>
Real xz_rect<frontFace>::pdf_value(const Point3 &origin,
                                   const Vector3 &v) const {
  HitRecord rec;
  if (!this->hit(Ray(origin, v), 0.001, infinity, rec))
    return 0;

  auto area = (x1 - x0) * (z1 - z0);
  auto distance_squared = rec.t * rec.t * v.length_sqr();
  auto cosine = fabs(v.dot(rec.normal) / v.length());

  return distance_squared / (cosine * area);
}
template <bool frontFace>
Vector3 xz_rect<frontFace>::random(const Point3 &origin) const {
  auto random_point = Point3({random_real(x0, x1), k, random_real(z0, z1)});
  return random_point - origin;
}
template <bool frontFace>
Real yz_rect<frontFace>::pdf_value(const Point3 &origin,
                                   const Vector3 &v) const {
  HitRecord rec;
  if (!this->hit(Ray(origin, v), 0.001, infinity, rec))
    return 0;

  auto area = (z1 - z0) * (y1 - y0);
  auto distance_squared = rec.t * rec.t * v.length_sqr();
  auto cosine = fabs(v.dot(rec.normal) / v.length());

  return distance_squared / (cosine * area);
}
template <bool frontFace>
Vector3 yz_rect<frontFace>::random(const Point3 &origin) const {
  auto random_point = Point3({k, random_real(y0, y1), random_real(z0, z1)});
  return random_point - origin;
}

template <bool frontFace> Photon xy_rect<frontFace>::random_photon() const {
  auto random_point = Point3({random_real(x0, x1), random_real(y0, y1), k});
  const Vector3 n{0, 0, frontFace ? 1 : -1};
  return gen_photon(mp, random_point, n);
}

template <bool frontFace> Photon xz_rect<frontFace>::random_photon() const {
  auto random_point = Point3({random_real(x0, x1), k, random_real(z0, z1)});
  const Vector3 n{0, frontFace ? 1 : -1, 0};
  return gen_photon(mp, random_point, n);
}

template <bool frontFace> Photon yz_rect<frontFace>::random_photon() const {
  auto random_point = Point3({k, random_real(y0, y1), random_real(z0, z1)});
  const Vector3 n{frontFace ? 1 : -1, 0, 0};
  return gen_photon(mp, random_point, n);
}

} // namespace rt
} // namespace dym
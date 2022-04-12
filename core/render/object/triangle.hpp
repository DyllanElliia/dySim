/*
 * @Author: DyllanElliia
 * @Date: 2022-04-11 14:30:56
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-12 14:54:54
 * @Description:
 */
/*
 * @Author: DyllanElliia
 * @Date: 2022-04-11 14:22:27
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-11 14:29:56
 * @Description:
 */
#pragma once

#include "../baseClass.hpp"

namespace dym {
namespace rt {
struct Vertex {
 public:
  Point3 point;
  Real u, v;
  Vector3 normal;

 public:
  Vertex() : point(0), u(0), v(0), normal(0) {}
  Vertex(const Point3& in_point, const Vector3& in_normal, const Real& in_u,
         const Real& in_v) {
    point = in_point;
    normal = in_normal;
    u = in_u, v = in_v;
  }
};

namespace {
Vertex emptyVer;
}

class Triangle : public Hittable {
 private:
  _DYM_FORCE_INLINE_ bool intersects(const Vector3& ray_origin,
                                     const Vector3& ray_dir,
                                     Real& out_t) const {
    const Real epsilon = 1e-7;
    auto edge1 = v1.point - v0.point;
    auto edge2 = v2.point - v0.point;
    auto h = dym::vector::cross(ray_dir, edge2);
    auto a = dym::vector::dot(edge1, h);
    if (a > -epsilon && a < epsilon) return false;
    auto f = 1 / a;
    auto s = ray_origin - v0.point;
    auto u = f * (dym::vector::dot(s, h));
    if (u < 0.0 || u > 1.0) return false;
    auto q = dym::vector::cross(s, edge1);
    auto v = f * dym::vector::dot(ray_dir, q);
    if (v < 0.0 || u + v > 1.0) return false;
    auto t = f * dym::vector::dot(edge2, q);
    if (t <= epsilon) return false;
    out_t = t;
    return true;
  }

  _DYM_FORCE_INLINE_ void get_triangle_uv(const Point3& p, Real& u,
                                          Real& v) const {
    auto f1 = v0.point - p;
    auto f2 = v1.point - p;
    auto f3 = v2.point - p;
    auto a =
        dym::vector::cross(v0.point - v1.point, v0.point - v2.point).length();
    auto a1 = dym::vector::cross(f2, f3).length() / a;
    auto a2 = dym::vector::cross(f3, f1).length() / a;
    auto a3 = dym::vector::cross(f1, f2).length() / a;
    u = v0.u * a1 + v1.u * a2 + v2.u * a3,
    v = v0.v * a1 + v1.v * a2 + v2.v * a3;
  }

 public:
  Triangle() : v0(emptyVer), v1(emptyVer), v2(emptyVer) {
    DYM_ERROR(
        "DYM::RT::TRIANGLE ERROR: v0, v1, v2 are Reference Variables!\nPlease "
        "check the constructor's input!");
  }
  Triangle(const Vertex& v0i, const Vertex& v1i, const Vertex& v2i,
           shared_ptr<Material> m)
      : v0(v0i),
        v1(v1i),
        v2(v2i),
        normal((v1i.point - v0i.point).cross(v2i.point - v0i.point)),
        mat_ptr(m){
            // qprint("normal", normal);
        };

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;
  virtual bool bounding_box(aabb& output_box) const override;

  virtual Real pdf_value(const Point3& origin, const Vector3& v) const override;
  virtual Vector3 random(const Point3& origin) const override;

 public:
  const Vertex &v0, &v1, &v2;
  Vector3 normal;
  shared_ptr<Material> mat_ptr;
};

bool Triangle::hit(const Ray& r, Real t_min, Real t_max, HitRecord& rec) const {
  Real intersect_t = 0;
  if (!intersects(r.origin(), r.direction(), intersect_t)) return false;
  if (intersect_t < t_min || t_max < intersect_t) return false;
  rec.t = intersect_t;
  rec.p = r.at(rec.t);
  rec.set_face_normal(r, normal.normalize());
  get_triangle_uv(rec.p, rec.u, rec.v);
  rec.mat_ptr = mat_ptr;
  return true;
}

bool Triangle::bounding_box(aabb& output_box) const {
  auto mi = dym::min(dym::min(v0.point, v1.point), v2.point);
  auto ma = dym::max(dym::max(v0.point, v1.point), v2.point);
  const auto epsilon = (ma - mi) / 10.0;
  output_box = aabb(mi - epsilon, ma + epsilon);
  return true;
}

Real Triangle::pdf_value(const Point3& origin, const Vector3& v) const {
  // qprint("here maybe");
  HitRecord rec;
  if (!this->hit(Ray(origin, v), 0.001, infinity, rec)) return 0;

  auto area = normal.length() / 2;
  auto distance_squared = rec.t * rec.t * v.length_sqr();
  auto cosine = fabs(v.dot(rec.normal) / v.length());

  return distance_squared / (cosine * area);
}
Vector3 Triangle::random(const Point3& origin) const {
  const auto sqrtR1 = dym::sqrt(random_real(0, 1));
  const auto R2 = random_real(0, 1);
  auto random_point = (1 - sqrtR1) * v0.point + sqrtR1 * (1 - R2) * v1.point +
                      sqrtR1 * R2 * v2.point;
  return random_point - origin;
}

}  // namespace rt
}  // namespace dym
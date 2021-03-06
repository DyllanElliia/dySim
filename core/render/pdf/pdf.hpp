/*
 * @Author: DyllanElliia
 * @Date: 2022-03-23 16:09:04
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-23 16:49:50
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
namespace dym {
namespace rt {
class pdf {
 public:
  virtual ~pdf() {}

  virtual Real value(const Vector3& direction) const = 0;
  virtual Vector3 generate() const = 0;
};

class cosine_pdf : public pdf {
 public:
  cosine_pdf(const Vector3& w) { uvw.build_from_w(w); }

  virtual Real value(const Vector3& direction) const override {
    auto cosine = direction.normalize().dot(uvw.w());
    return (cosine <= 0) ? 0 : cosine / pi;
  }

  virtual Vector3 generate() const override {
    return uvw.local(random_cosine_direction());
  }

 public:
  onb uvw;
};

class hittable_pdf : public pdf {
 public:
  hittable_pdf(shared_ptr<Hittable> p, const Point3& origin)
      : ptr(p), o(origin) {}

  virtual Real value(const Vector3& direction) const override {
    return ptr->pdf_value(o, direction);
  }

  virtual Vector3 generate() const override { return ptr->random(o); }

 public:
  Point3 o;
  shared_ptr<Hittable> ptr;
};

class mixture_pdf : public pdf {
 public:
  mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1) {
    p[0] = p0;
    p[1] = p1;
  }

  virtual Real value(const Vector3& direction) const override {
    return 0.5 * p[0]->value(direction) + 0.5 * p[1]->value(direction);
  }

  virtual Vector3 generate() const override {
    if (random_real() < 0.5)
      return p[0]->generate();
    else
      return p[1]->generate();
  }

 public:
  shared_ptr<pdf> p[2];
};
}  // namespace rt
}  // namespace dym
/*
 * @Author: DyllanElliia
 * @Date: 2022-03-23 16:09:04
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-03-23 16:49:50
 * @Description:
 */
#pragma once
#include "../baseClass.hpp"
#include "dyMath.hpp"
#include "math/define.hpp"
namespace dym {
namespace rt {
class pdf {
public:
  virtual ~pdf() {}

  virtual Real value(const Vector3 &direction) const = 0;
  virtual Vector3 generate() const = 0;
};

class cosine_pdf : public pdf {
private:
  _DYM_FORCE_INLINE_ Vector3 random_cosine_direction() const {
    auto r1 = random_real();
    auto r2 = random_real();
    auto z = sqrt(1 - r2);

    auto phi = 2 * pi * r1;
    auto x = cos(phi) * sqrt(r2);
    auto y = sin(phi) * sqrt(r2);

    return Vector3({x, y, z});
  }

public:
  cosine_pdf(const Vector3 &w) { uvw.build_from_w(w); }

  virtual Real value(const Vector3 &direction) const override {
    auto cosine = direction.normalize().dot(uvw.w());
    return (cosine <= 0) ? 0 : cosine / pi;
  }

  virtual Vector3 generate() const override {
    return uvw.local(random_cosine_direction());
  }

public:
  onb uvw;
};

class GTR2_pdf : public pdf {
private:
  _DYM_FORCE_INLINE_ Vector3 random_GTR2_direction(const Real &xi_1,
                                                   const Real &xi_2,
                                                   const Real &alpha) const {
    Real phi_h = phi_h = 2.0 * pi * xi_1;
    Real sin_phi_h = sin(phi_h);
    Real cos_phi_h = cos(phi_h);

    Real cos_theta_h =
        sqrt((1.0 - xi_2) / (1.0 + (alpha * alpha - 1.0) * xi_2));
    Real sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));
    return Vector3(
        {sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h});
  }

  _DYM_FORCE_INLINE_ Real solve_GTR2_pdf(const Real &NdotH,
                                         const Real &alpha) const {
    Real a2 = alpha * alpha;
    Real t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
    return a2 / (pi * t * t);
  }

public:
  GTR2_pdf(const Vector3 &w, const Real &fuzz, const Vector3 &dir_in)
      : alpha((fuzz)), alpha_i(sqr(1 - fuzz)), dir_in(dir_in) {
    uvw.build_from_w(w);
  }

  virtual Real value(const Vector3 &direction) const override {
    Vector3 H = (direction - dir_in).normalize();
    Real NdotH = uvw.w().dot(H);
    Real Ds = solve_GTR2_pdf(NdotH, 1 - alpha);
    // return Ds * NdotH / (4.0 * direction.dot(H));
    return Ds * NdotH / (direction.dot(H));
  }

  virtual Vector3 generate() const override {
    return (dir_in).reflect(
        uvw.local(random_GTR2_direction(random_real(), random_real(), alpha)));
  }

public:
  onb uvw;
  Real alpha, alpha_i;
  Vector3 dir_in;
};

class hittable_pdf : public pdf {
public:
  hittable_pdf(shared_ptr<Hittable> p, const Point3 &origin)
      : ptr(p), o(origin) {}

  virtual Real value(const Vector3 &direction) const override {
    return ptr->pdf_value(o, direction.normalize());
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

  virtual Real value(const Vector3 &direction) const override {
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
} // namespace rt
} // namespace dym
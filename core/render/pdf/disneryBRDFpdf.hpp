#pragma once

#include "./pdf.hpp"
#include "render/randomFun.hpp"

namespace dym {
namespace rt {
struct disBrdfMatPix {
  ColorRGB baseColor;
  Real subSurface;
  Real metallic;
  Real specular;
  Real specularTint;
  Real roughness;
  Real anisotropic;
  Real sheen;
  Real sheenTint;
  Real clearcoat;
  Real clearcoatGloss;
};
class disnery_pdf : public pdf {
public:
  disnery_pdf(const Vector3 &w, const Vector3 &dir_in, const disBrdfMatPix &mat)
      : mat(mat), dir_in_i(-dir_in) {
    uvw.build_from_w(w);
  }

  virtual Real value(const Vector3 &direction) const override {
    auto N = uvw.w();
    auto &V = dir_in_i, &L = direction;
    Real NdotL = N.dot(L);
    Real NdotV = N.dot(V);

    if (NdotL < 0 || NdotV < 0)
      return 0;

    Vector3 H = (L + V).normalize();
    Real NdotH = N.dot(H);
    Real LdotH = L.dot(H);

    // specualr
    Real alpha_GTR1 = lerp(0.1, 0.001, mat.clearcoatGloss);
    Real alpha_GTR2 = max(0.001, sqr(0.5 + mat.roughness / 2.0));
    Real Ds = solve_GTR2_pdf(NdotH, alpha_GTR2);
    Real Dr = solve_GTR1_pdf(NdotH, alpha_GTR1);

    // cal pdf
    Real pdf_diff = NdotL / Pi;
    Real pdf_spec = Ds * NdotH / (4 * LdotH);
    Real pdf_clco = Dr * NdotH / (4 * LdotH);

    // Sum Radiancy degree
    Real r_diff = (1 - mat.metallic);
    Real r_spec = mat.specular;
    Real r_clco = 0.25 * mat.clearcoat;
    Real r_sum = r_diff + r_spec + r_clco;

    // According to radiancy, cal probability
    Real p_diff = r_diff / r_sum;
    Real p_spec = r_spec / r_sum;
    Real p_clco = r_clco / r_sum;

    Real pdf = p_diff * pdf_diff + p_spec * pdf_spec + p_clco * pdf_clco;

    return clamp(pdf, 1e-4, 1.0);
  }

  virtual Vector3 generate() const override {
    Real alpha_GTR1 = lerp(0.1, 0.001, mat.clearcoatGloss);
    Real alpha_GTR2 = max(0.001, sqr(mat.roughness));

    // Sum Radiancy degree
    Real r_diff = (1 - mat.metallic);
    Real r_spec = mat.specular;
    Real r_clco = 0.25 * mat.clearcoat;
    Real r_sum = r_diff + r_spec + r_clco;

    // According to radiancy, cal probability
    Real p_diff = r_diff / r_sum;
    Real p_spec = r_spec / r_sum;
    Real p_clco = r_clco / r_sum;

    // sample
    Real rd = random_real();
    if (rd <= p_diff)
      return uvw.local(random_cosine_direction());
    if (rd <= p_diff + p_spec)
      return uvw.local(
          random_GTR2_direction(random_real(), random_real(), alpha_GTR2));
    // p_dirr + p_spec < rd
    return uvw.local(
        random_GTR1_direction(random_real(), random_real(), alpha_GTR1));
  }

public:
  onb uvw;
  Vector3 dir_in_i;
  disBrdfMatPix mat;
};
} // namespace rt
} // namespace dym
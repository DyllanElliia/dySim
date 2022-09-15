#pragma once
#include "../baseClass.hpp"
#include "render/pdf/disneryBRDFpdf.hpp"
#include "render/randomFun.hpp"
#include <memory>

namespace dym {
namespace rt {
struct DisneryMat {
  shared_ptr<Texture> baseColor;
  shared_ptr<Texture> subSurface;
  shared_ptr<Texture> metallic;
  shared_ptr<Texture> specualr;
  shared_ptr<Texture> specualrTint;
  shared_ptr<Texture> roughness;
  shared_ptr<Texture> anisotropic;
  shared_ptr<Texture> sheen;
  shared_ptr<Texture> sheenTint;
  shared_ptr<Texture> clearcoat;
  shared_ptr<Texture> clearcoatGloss;
};

class DisneryBRDF : public Material {
public:
  DisneryBRDF(const DisneryMat &disneryMaterial) : mat(disneryMaterial) {}

  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const override {
    disBrdfMatPix pdfmat;

    // TODO: fix pdfmat

    Vector3 reflected = r_in.direction().normalize().reflect(rec.normal);
    srec.specular_ray = Ray(rec.p, reflected);
    srec.attenuation = mat.baseColor->value(rec.u, rec.v, rec.p);
    srec.is_specular = false;
    srec.pdf_ptr =
        make_shared<disnery_pdf>(rec.normal, r_in.direction(), pdfmat);

    return true;
  }

  virtual Vector3 BRDF_Evaluate(const Ray &r_in, const Ray &scattered,
                                const HitRecord &rec,
                                const ScatterRecord &srec) const {
    Vector3 L = scattered.direction(), V = -r_in.direction();
    Vector3 N = rec.normal;
    Vector3 H = (L + V).normalize();
    auto NdotH = N.dot(H);
    auto LdotH = L.dot(H);
    auto NdotL = N.dot(L);
    auto NdotV = N.dot(V);
    auto alphaG = sqr(0.5 + fuzz / 2.0);
    auto Ds = solve_GTR2_pdf(NdotH, alphaG);
    auto FH = SchlickFresnel(LdotH);
    auto Gs = smithG_GGX(NdotV, alphaG) * smithG_GGX(NdotL, alphaG);

    ColorRGB Fs = lerp(srec.attenuation, ColorRGB(1.0), FH);
    if (fuzz < 0.01)
      return Fs;

    return Gs * Ds * Fs;
    // return Gs * Ds;
    // return 1;
  }

private:
  _DYM_FORCE_INLINE_ Real SchlickFresnel(const Real &u) const {
    Real m = clamp(1 - u, 0, 1);
    Real m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
  }

  _DYM_FORCE_INLINE_ Real smithG_GGX(const Real &NdotV,
                                     const Real &alphaG) const {
    Real a = alphaG * alphaG;
    Real b = NdotV * NdotV;
    return NdotV / (NdotV + sqrt(a + b - a * b));
  }

public:
  DisneryMat mat;
};
} // namespace rt
} // namespace dym
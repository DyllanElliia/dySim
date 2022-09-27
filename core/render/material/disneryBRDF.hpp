#pragma once
#include "../baseClass.hpp"
#include "render/pdf/disneryBRDFpdf.hpp"
#include "render/randomFun.hpp"
#include <any>
#include <memory>

namespace dym {
namespace rt {
struct DisneryMat {
  shared_ptr<Texture> baseColor;
  shared_ptr<Texture> subSurface;
  shared_ptr<Texture> metallic;
  shared_ptr<Texture> specular;
  shared_ptr<Texture> specularTint;
  shared_ptr<Texture> roughness;
  shared_ptr<Texture> anisotropic;
  shared_ptr<Texture> sheen;
  shared_ptr<Texture> sheenTint;
  shared_ptr<Texture> clearcoat;
  shared_ptr<Texture> clearcoatGloss;
  shared_ptr<Texture> lightEmit;
  DisneryMat() { lightEmit = std::make_shared<SolidColor>(0.); }
};

class DisneryBRDF : public Material {
public:
  DisneryBRDF(const DisneryMat &disneryMaterial) : mat(disneryMaterial) {}

  virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                       ScatterRecord &srec) const override {
    disBrdfMatPix pdfmat;

    // TODO: fix pdfmat
    pdfmat.baseColor = mat.baseColor->value(rec.u, rec.v, rec.p);
    pdfmat.subSurface = mat.subSurface->value(rec.u, rec.v, rec.p)[0];
    pdfmat.metallic = mat.metallic->value(rec.u, rec.v, rec.p)[0];
    pdfmat.specular = mat.specular->value(rec.u, rec.v, rec.p)[0];
    pdfmat.specularTint = mat.specularTint->value(rec.u, rec.v, rec.p)[0];
    pdfmat.roughness = mat.roughness->value(rec.u, rec.v, rec.p)[0];
    pdfmat.anisotropic = mat.anisotropic->value(rec.u, rec.v, rec.p)[0];
    pdfmat.sheen = mat.sheen->value(rec.u, rec.v, rec.p)[0];
    pdfmat.sheenTint = mat.sheenTint->value(rec.u, rec.v, rec.p)[0];
    pdfmat.clearcoat = mat.clearcoat->value(rec.u, rec.v, rec.p)[0];
    pdfmat.clearcoatGloss = mat.clearcoatGloss->value(rec.u, rec.v, rec.p)[0];

    Vector3 reflected = r_in.direction().normalize().reflect(rec.normal);
    srec.specular_ray = Ray(rec.p, reflected);
    srec.attenuation = pdfmat.baseColor;
    srec.is_specular = pdfmat.metallic;
    srec.pdf_ptr =
        make_shared<disnery_pdf>(rec.normal, r_in.direction(), pdfmat);
    srec.otherData = pdfmat;

    return true;
  }

  virtual Vector3 BRDF_Evaluate(const Ray &r_in, const Ray &scattered,
                                const HitRecord &rec,
                                const ScatterRecord &srec) const override {
    const disBrdfMatPix pdfmat = std::any_cast<disBrdfMatPix>(srec.otherData);
    Vector3 L = scattered.direction(), V = -r_in.direction();
    Vector3 N = rec.normal;
    auto NdotL = N.dot(L);
    auto NdotV = N.dot(V);
    if (NdotL < 0 || NdotV < 0)
      return 0.;
    Vector3 H = (L + V).normalize();
    auto NdotH = N.dot(H);
    auto LdotH = L.dot(H);

    // cal color
    auto Cdlin = mon2lin(pdfmat.baseColor);
    Real Cdlum = 0.3 * Cdlin[0] + 0.6 * Cdlin[1] + 0.1 * Cdlin[2];
    Vector3 Ctint = Cdlum > 0. ? Cdlin / Cdlum : 1.;
    Vector3 Cspec =
        pdfmat.specular * .08 * lerp(Vector3(1.), Ctint, pdfmat.specularTint);
    Vector3 Cspec0 = lerp(Cspec, Cdlin, pdfmat.metallic);
    // Vector3 Cspec0 = 0.0;
    Vector3 Csheen = lerp(Vector3(1.), Ctint, pdfmat.sheenTint);

    // cal global value
    Real FL = SchlickFresnel(NdotL);
    Real FV = SchlickFresnel(NdotV);

    // diffuse
    Real Fd90 = .5 + 2. * sqr(LdotH) * pdfmat.roughness;
    Real Fd = lerp(1., Fd90, FL) * lerp(1., Fd90, FV);

    // subSurface diffuse
    Real Fss90 = sqr(LdotH) * pdfmat.roughness;
    Real Fss = lerp(1., Fss90, FL) * lerp(1., Fss90, FV);
    Real ss = 1.25 * (Fss * (1. / (NdotL + NdotV) - .5) + .5);

    // specular - isotropic
    // Real alpha = max(0.001, sqr(pdfmat.roughness));
    auto alphaG = sqr(0.5 + pdfmat.roughness / 2.0);
    auto Ds = solve_GTR2_pdf(NdotH, alphaG);
    auto FH = SchlickFresnel(LdotH);
    auto Gs = smithG_GGX(NdotV, alphaG) * smithG_GGX(NdotL, alphaG);
    ColorRGB Fs = lerp(Cspec0, ColorRGB(1.), FH);

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    Real Dr = solve_GTR1_pdf(NdotH, lerp(0.1, 0.001, pdfmat.clearcoatGloss));
    Real Fr = lerp(0.04, 1.0, FH);
    Real Gr = smithG_GGX(NdotV, 0.25) * smithG_GGX(NdotL, 0.25);

    // sheen
    ColorRGB Fsheen = FH * pdfmat.sheen * Csheen;

    // Return: cal Fr
    auto diffuse = (1.0 / Pi) * mix(Fd, ss, pdfmat.subSurface) * Cdlin + Fsheen;
    // auto diffuse = 0.0;
    auto specular = Gs * Fs * Ds;
    auto clearcoat = 0.25 * Gr * Fr * Dr * pdfmat.clearcoat;
    // auto clearcoat = 0.0;

    return diffuse * (1. - pdfmat.metallic) + specular + clearcoat;
    // return Cspec0;
  }

  virtual ColorRGB emitted(const Ray &r_in, const HitRecord &rec, Real u,
                           Real v, const Point3 &p) const {
    return mat.lightEmit->value(rec.u, rec.v, rec.p);
  }

private:
  _DYM_FORCE_INLINE_ Vector3 mon2lin(const Vector3 &x) const {
    return {pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2)};
  }
  _DYM_FORCE_INLINE_ Vector3 mix(const Real &a, const Real &b,
                                 const Vector3 &v) const {
    return Vector3([&](Real &o, int i) { o = lerp(a, b, v[i]); });
  }

public:
  DisneryMat mat;
};
} // namespace rt
} // namespace dym
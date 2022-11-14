/*
 * @Author: DyllanElliia
 * @Date: 2022-03-01 15:34:03
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-19 13:50:24
 * @Description:
 */
#include "render/object/sphere.hpp"
#include "render/randomFun.hpp"
#include <dyGraphic.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>

namespace dym {
namespace rt {

class disnery_aniso_pdf : public pdf {
public:
  disnery_aniso_pdf(const Vector3 &w, const Vector3 &dir_in,
                    const disBrdfMatPix &mat)
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
    Real alpha_GTR2 = sqr(0.5 + mat.roughness / 2.0);
    Real aspect = sqrt(1.0 - mat.anisotropic * 0.9);
    Real ax = max(0.001, alpha_GTR2 / aspect);
    Real ay = max(0.001, alpha_GTR2 * aspect);
    Vector3 X = uvw.v(), Y = uvw.u();
    // Real Ds = solve_GTR2_pdf(NdotH, alpha_GTR2);
    Real Ds = solve_GTR2_aniso_pdf(NdotH, H.dot(X), H.dot(Y), ax, ay);
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
    Real alpha_GTR2 = sqr(0.5 + mat.roughness / 2.0);

    // Sum Radiancy degree
    Real r_diff = (1 - mat.metallic);
    Real r_spec = mat.specular;
    Real r_clco = 0.25 * mat.clearcoat;
    Real r_sum = r_diff + r_spec + r_clco;

    // According to radiancy, cal probability
    Real p_diff = r_diff / r_sum;
    Real p_spec = r_spec / r_sum;
    Real p_clco = r_clco / r_sum;

    Real aspect = sqrt(1.0 - mat.anisotropic * 0.9);
    Real ax = max(0.001, alpha_GTR2 / aspect);
    Real ay = max(0.001, alpha_GTR2 * aspect);
    Vector3 X = uvw.v(), Y = uvw.u();
    // sample
    Real rd = random_real();
    if (rd <= p_diff)
      return uvw.local(random_cosine_direction());
    if (rd <= p_diff + p_spec) {
      auto gtr2 =
          random_GTR2_direction(random_real(), random_real(), alpha_GTR2);
      gtr2[0] /= aspect, gtr2[1] *= aspect;
      return uvw.local(gtr2);
    }
    // p_dirr + p_spec < rd
    return uvw.local(
        random_GTR1_direction(random_real(), random_real(), alpha_GTR1));
  }

public:
  onb uvw;
  Vector3 dir_in_i;
  disBrdfMatPix mat;
};
class DisneryBRDF_aniso : public Material {
public:
  DisneryBRDF_aniso(const DisneryMat &disneryMaterial) : mat(disneryMaterial) {}

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
        make_shared<disnery_aniso_pdf>(rec.normal, r_in.direction(), pdfmat);
    srec.otherData = pdfmat;

    return true;
  }

  virtual Vector3 BxDF_Evaluate(const Ray &r_in, const Ray &scattered,
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
    auto alphaG = sqr(0.5 + pdfmat.roughness / 2.0);

    // auto Ds = solve_GTR2_pdf(NdotH, alphaG);
    // auto FH = SchlickFresnel(LdotH);
    // auto Gs = smithG_GGX(NdotV, alphaG) * smithG_GGX(NdotL, alphaG);
    // ColorRGB Fs = lerp(Cspec0, ColorRGB(1.), FH);
    onb uvw;
    uvw.build_from_w(N);
    Vector3 X = uvw.v(), Y = uvw.u();
    Real aspect = sqrt(1.0 - pdfmat.anisotropic * 0.9);
    Real ax = max(0.001, alphaG / aspect);
    Real ay = max(0.001, alphaG * aspect);
    Real Ds = solve_GTR2_aniso_pdf(NdotH, H.dot(X), H.dot(Y), ax, ay);
    Real FH = SchlickFresnel(LdotH);
    Vector3 Fs = lerp(Cspec0, Vector3(1.), FH);
    Real Gs;
    Gs = smithG_GGX_aniso(NdotL, L.dot(X), L.dot(Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, V.dot(X), V.dot(Y), ax, ay);

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
    // return specular;
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

_DYM_FORCE_INLINE_ auto whiteMetalSur(Real objcolor, Real fuzz = 0) {
  auto white_surface =
      std::make_shared<dym::rt::Metal>(dym::rt::ColorRGB(objcolor), fuzz);

  return white_surface;
}

using SolidColor = dym::rt::SolidColor;
using ImageTexture = dym::rt::ImageTexture<3>;
_DYM_FORCE_INLINE_ auto disneryBRDF_subsurface(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3(0.75));
  dmat.subSurface = std::make_shared<SolidColor>(x);
  dmat.metallic = std::make_shared<SolidColor>(.01);
  dmat.specular = std::make_shared<SolidColor>(.01);
  dmat.specularTint = std::make_shared<SolidColor>(.1);
  dmat.roughness = std::make_shared<SolidColor>(.8);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(.1);
  dmat.sheenTint = std::make_shared<SolidColor>(.1);
  dmat.clearcoat = std::make_shared<SolidColor>(.1);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}
_DYM_FORCE_INLINE_ auto disneryBRDF_metal(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3{0.9, 0.75, 0.2});
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(dym::min(0.95, x));
  dmat.specular = std::make_shared<SolidColor>(.9);
  dmat.specularTint = std::make_shared<SolidColor>(.2);
  dmat.roughness = std::make_shared<SolidColor>(.1);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}
_DYM_FORCE_INLINE_ auto disneryBRDF_specular(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3{0.9, 0.75, 0.2});
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(0.5);
  dmat.specular = std::make_shared<SolidColor>(x);
  dmat.specularTint = std::make_shared<SolidColor>(.0);
  dmat.roughness = std::make_shared<SolidColor>(.1);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}
_DYM_FORCE_INLINE_ auto disneryBRDF_specularTint(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor =
      std::make_shared<SolidColor>(dym::Vector3{0.9, 0.75, 0.2} / 2.);
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(0.01);
  dmat.specular = std::make_shared<SolidColor>(0.99);
  dmat.specularTint = std::make_shared<SolidColor>(x);
  dmat.roughness = std::make_shared<SolidColor>(.1);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

_DYM_FORCE_INLINE_ auto disneryBRDF_roughness(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3{0.9, 0.75, 0.2});
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(0.95);
  dmat.specular = std::make_shared<SolidColor>(.99);
  dmat.specularTint = std::make_shared<SolidColor>(.1);
  dmat.roughness = std::make_shared<SolidColor>(dym::lerp(0.0, 0.9, x));
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

_DYM_FORCE_INLINE_ auto disneryBRDF_sheen(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3{0.08, 0.18, 0.45});
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(0.01);
  dmat.specular = std::make_shared<SolidColor>(.01);
  dmat.specularTint = std::make_shared<SolidColor>(.1);
  dmat.roughness = std::make_shared<SolidColor>(0.85);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(x);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}
_DYM_FORCE_INLINE_ auto disneryBRDF_sheenTint(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3{0.08, 0.18, 0.45});
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(0.01);
  dmat.specular = std::make_shared<SolidColor>(.01);
  dmat.specularTint = std::make_shared<SolidColor>(.1);
  dmat.roughness = std::make_shared<SolidColor>(0.85);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(1.);
  dmat.sheenTint = std::make_shared<SolidColor>(x);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

_DYM_FORCE_INLINE_ auto disneryBRDF_clearcoat(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3{0.08, 0.18, 0.45});
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(0.01);
  dmat.specular = std::make_shared<SolidColor>(.01);
  dmat.specularTint = std::make_shared<SolidColor>(.1);
  dmat.roughness = std::make_shared<SolidColor>(0.85);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(x);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(1.);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

_DYM_FORCE_INLINE_ auto disneryBRDF_clearcoatGloss(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3{0.08, 0.18, 0.45});
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(0.01);
  dmat.specular = std::make_shared<SolidColor>(.01);
  dmat.specularTint = std::make_shared<SolidColor>(.1);
  dmat.roughness = std::make_shared<SolidColor>(0.85);
  dmat.anisotropic = std::make_shared<SolidColor>(.0);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.9);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(x);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF>(dmat);

  return brdf_surface;
}

_DYM_FORCE_INLINE_ auto disneryBRDF_anisotropic(Real x) {
  dym::rt::DisneryMat dmat;

  dmat.baseColor = std::make_shared<SolidColor>(dym::Vector3{0.9, 0.75, 0.2});
  dmat.subSurface = std::make_shared<SolidColor>(0.01);
  dmat.metallic = std::make_shared<SolidColor>(0.85);
  dmat.specular = std::make_shared<SolidColor>(.9);
  dmat.specularTint = std::make_shared<SolidColor>(0.0);
  dmat.roughness = std::make_shared<SolidColor>(.1);
  dmat.anisotropic = std::make_shared<SolidColor>(x);
  dmat.sheen = std::make_shared<SolidColor>(0.);
  dmat.sheenTint = std::make_shared<SolidColor>(0.);
  dmat.clearcoat = std::make_shared<SolidColor>(0.);
  dmat.clearcoatGloss = std::make_shared<SolidColor>(.6);

  auto brdf_surface = std::make_shared<dym::rt::DisneryBRDF_aniso>(dmat);

  return brdf_surface;
}

auto cornell_box() {
  dym::rt::HittableList objects;
  Real fuzz = 0.2;
  auto red =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({.65, .05, .05}));
  auto white =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({.73, .73, .73}));
  auto green =
      std::make_shared<dym::rt::Lambertian>(dym::rt::ColorRGB({.12, .45, .15}));
  auto light = std::make_shared<dym::rt::DiffuseLight>(dym::rt::ColorRGB(20));

  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 1, green));
  objects.add(std::make_shared<dym::rt::yz_rect>(0, 1, 0, 1, 0, red));

  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 0, white));
  objects.add(std::make_shared<dym::rt::xz_rect>(0, 1, 0, 1, 1, white));
  objects.add(std::make_shared<dym::rt::xy_rect>(0, 1, 0, 1, 1, white));
  objects.add(std::make_shared<dym::rt::xy_rect>(0, 1, 0, 1, 0, white));

  Real begin = 0.35, end = 0.65;
  objects.add(
      std::make_shared<dym::rt::xz_rect>(begin, end, begin, end, 0.998, light));

  return dym::rt::BvhNode(objects);
}

int main(int argc, char const *argv[]) {
  const auto aspect_ratio = 2000. / 200.;
  const int image_width = 2000;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  int samples_per_pixel = 50;
  const int max_depth = 50;
  dym::Tensor<dym::Vector<Real, dym::PIC_RGB>> image(
      0, dym::gi(image_height, image_width));
  dym::Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>> imageP(
      0, dym::gi(image_height, image_width));
  // World

  dym::rt::HittableList world;
  dym::rt::HittableList lights;
  Real begin = 0.35, end = 0.65;
  lights.add(std::make_shared<dym::rt::xz_rect>(
      begin, end, begin, end, 0.998, std::shared_ptr<dym::rt::Material>()));

  world.add(std::make_shared<dym::rt::BvhNode>(cornell_box()));

  for (Real x = 0.1; x < 0.9; x += 0.08)
  // for (Real y = 0.1; y < 0.9; y += 0.08)
  {
    world.addObject<dym::rt::Sphere>(dym::rt::Point3({x, 0.03, 0.5}), 0.03,
                                     disneryBRDF_clearcoat((x - 0.1) / 0.8));
    // world.addObject<dym::rt::Sphere>(
    //     dym::rt::Point3({x, 0.15, y}), 0.1,
    //     whiteMetalSur(1 - y, lerp(0.0, 0.05, (x - 0.15))));
    // qprint(x, y);
  }

  dym::rt::RtRender render(image_width, image_height);

  // Camera
  dym::rt::Point3 lookfrom({0.5, .2, 0.0});
  dym::rt::Point3 lookat({0.5, 0.03, 0.5});
  dym::Vector3 vup({0, 1, 1});
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 2.0;
  render.cam.setCamera(lookfrom, lookat, vup, 10, aspect_ratio, aperture,
                       dist_to_focus);

  // dym::rt::Point3 lookfrom({0.5, 0.5, -1.35});
  // dym::rt::Point3 lookat({0.5, 0.5, 0});
  // dym::Vector3 vup({0, 1, 0});
  // auto dist_to_focus = (lookfrom - lookat).length();
  // auto aperture = 2.0;
  // render.cam.setCamera(lookfrom, lookat, vup, 40, aspect_ratio, aperture,
  //                      dist_to_focus);

  render.worlds.addObject<dym::rt::BvhNode>(world);
  render.lights = lights;

  // GUI
  dym::GUI gui("rt");
  gui.init(image_width, image_height);
  Real t = 0.4, t_inv = 1 - t;
  dym::TimeLog time;
  int ccc = 1;

  time.reStart();
  gui.update([&]() {
    dym::TimeLog partTime;
    render.render(samples_per_pixel, max_depth);
    render.denoise();
    ccc++;
    time.record();
    time.reStart();
    auto image = render.getFrame();
    dym::imwrite(image, "./rt_out/sample/disnery/frame_" +
                            std::to_string(ccc - 1) + ".jpg");
    gui.imshow(image);
  });
  return 0;
}

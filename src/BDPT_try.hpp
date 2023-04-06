
#include "math/define.hpp"
#include "render/randomFun.hpp"
#include "render/ray.hpp"
#include <dyGraphic.hpp>
#include <dyMath.hpp>
#include <dyPicture.hpp>
#include <dyRender.hpp>

namespace dym {
namespace rt {

class BDPT_kernel : public RKernel {
public:
  virtual ColorRGB render(
      const Ray &r, const Hittable &world, shared_ptr<HittableList> lights,
      Real nST,
      const std::function<ColorRGB(const Ray &r)> &background =
          [](const Ray &r) { return ColorRGB(0.f); }) override {
    auto LightSubPath = generateLight(world, lights, nST);
    auto ViewSubPath = generateView(r, world, lights, nST);
    const int nE = min(ViewSubPath.n, nST), nL = min(LightSubPath.n, nST);
    for (int s = 0; s < nL; ++s)
      for (int t = 0; t < nE; ++t) {
        auto st = s + t;
      }
  }

  virtual bool render_preProcessing(RtMessage &rm) override {
    image = rm.image, camera = rm.cam;
    image_0.reShape(image.shape());
    image_0 = .0;
    return true;
  }
  virtual bool render_postProcessing(RtMessage &rm) override { return true; }

private:
  Camera camera;
  Tensor<dym::Vector<Real, dym::PIC_RGB>> image;
  Tensor<dym::Vector<Real, dym::PIC_RGB>> image_0;

  struct SubNode {
    Ray r;
    HitRecord rec;
    shared_ptr<pdf> matPdf;
    ColorRGB Le = -1;
  };
  _DYM_FORCE_INLINE_ SubNode pkgNode(const Ray &r, const HitRecord &rec,
                                     shared_ptr<pdf> matPdf) {
    SubNode n;
    n.r = r, n.rec = rec, n.matPdf = matPdf;
    return n;
  }
  struct SubPath {
    std::vector<SubNode> subPath;
    int n;
    SubPath() : n(0) {}
    int push_back(const SubNode &node) {
      subPath.push_back(node);
      return n++;
    }
    SubNode &back() { return subPath[n - 1]; }
    SubNode &operator[](const int &i) { return subPath[i]; }
    int length() { return subPath.size(); }
  };

  SubPath generateLight(const Hittable &world, shared_ptr<HittableList> lights,
                        int nL) {
    SubPath sb;
    // * random a photon from lights List.
    auto int_size = static_cast<int>(lights->objects.size());
    auto &lightObjPtr =
        lights->objects[static_cast<int>(random_real(0, int_size - 1))];
    auto lightPhoton = lightObjPtr->random_photon();
    // * record the photon HitRecord message.
    HitRecord lrec;
    if (!lightObjPtr->hit(Ray(lightPhoton.r.orig + 1e-2 * lightPhoton.r.dir,
                              -lightPhoton.r.dir),
                          0., infinity, lrec))
      return sb;
    sb.push_back(pkgNode(lightPhoton.r, lrec, nullptr));
    // * start to gen light SubPath
    Ray r = lightPhoton.r;
    while (sb.n < nL) {
      // * Can r hit next obj?
      HitRecord rec;
      if (!world.hit(r, 0.001, infinity, rec))
        break;
      // * Yes! But, Can this obj's Material reflect the light?
      ScatterRecord srec;
      if (!rec.mat_ptr->scatter(r, rec, srec)) {
        sb.push_back(pkgNode(r, rec, nullptr));
        break;
      }
      // * Wow! Can this obj's Material reflact any light?
      if (srec.is_specular && !srec.pdf_ptr) {
        sb.push_back(pkgNode(r, rec, nullptr));
        r = srec.specular_ray;
        continue;
      }
      // * Finally, pkg the node include mixPdf method to node.
      sb.push_back(pkgNode(r, rec, srec.pdf_ptr));
      shared_ptr<pdf> matPdf = srec.pdf_ptr;
      r = Ray(rec.p, matPdf->generate(), r.time());
    }
    return sb;
  }
  SubPath generateView(const Ray &r, const Hittable &world,
                       shared_ptr<HittableList> lights, int nE) {
    SubPath sb;
    // * Save camera photon.

    // * start to gen view SubPath
    while (sb.n < nE) {
      // * Can r hit next obj?

      // * Yes! But, Can this obj's Material reflect the light?

      // * Wow! Can this obj's Material reflact any light?

      // * Finally, pkg the node include mixPdf method to node.
    }
    return sb;
  }
};
} // namespace rt
} // namespace dym
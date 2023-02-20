
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
      Real RR,
      const std::function<ColorRGB(const Ray &r)> &background =
          [](const Ray &r) { return ColorRGB(0.f); }) override {}

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
  };
  struct SubPath {
    std::vector<SubNode> subPath;
    int n;
    SubPath() : n(0) {}
    int push_back(const SubNode &node) {
      subPath.push_back(node);
      return n++;
    }
    SubNode &operator[](const int &i) { return subPath[i]; }
    int length() { return subPath.size(); }
  };
};
} // namespace rt
} // namespace dym
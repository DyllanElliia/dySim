#pragma once

#include "../../../dyGraphic.hpp"
#include "../../../dyMath.hpp"

namespace dym {
namespace rdo {
class renderObject {
public:
  std::string name;

  renderObject(const std::string &name = "renderObject") : name(name) {}
  ~renderObject(){};
  virtual void Draw(rdt::Shader &shader, unsigned int instancedNum = 1) {}
};
} // namespace rdo
} // namespace dym

// object
#include "./box.hpp"
#pragma once

#include "./dual_alg.hpp"

namespace dym {
namespace AD {
template <typename Type1, typename Type2> class AdNode {
public:
  AdNode(Type1 lo, Type2 ro) {}

private:
  Type1 lObj;
  Type2 rObj;
};
} // namespace AD
} // namespace dym
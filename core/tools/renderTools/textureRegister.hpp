#pragma once

#include "../../dyGraphic.hpp"
#include "../str_hash.hpp"

#include <unordered_map>

namespace dym {
namespace rdt {

typedef unsigned int _TextureID_;
typedef unsigned int _ShaderID_;
typedef std::string _TextureName_;
typedef hash_t _TextureNameHash_;
namespace {
std::unordered_map<_ShaderID_, std::map<_TextureNameHash_, _TextureID_>>
    textureIdRegisterTable;
}

_TextureID_ texRegFind(_ShaderID_ id, _TextureName_ textureName) {
  // std::string key = std::to_string(id) + ":" + textureName;
  std::map<_TextureNameHash_, _TextureID_> &texN2texId =
      textureIdRegisterTable[id];
  _TextureNameHash_ textureNh = hash_(textureName.c_str());
  auto findRes = texN2texId.find(textureNh);
  if (findRes == texN2texId.end()) {
    _TextureID_ result;
    result = texN2texId.size();
    texN2texId[textureNh] = result;
    return result;
  }
  return findRes->second;
}

} // namespace rdt
} // namespace dym

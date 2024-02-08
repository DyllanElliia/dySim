#pragma once

#include "./modelLoader.hpp"
#include "math/define.hpp"

namespace dym {
namespace rdt {

namespace {
struct _texType_ {
  std::string diffuse = "texture_diffuse";
  std::string specular = "texture_specular";
  std::string normal = "texture_normal";
  std::string height = "texture_height";
};
}  // namespace
_texType_ TexType;

class Texture {
 public:
  Texture() {}
  Texture(const std::string &path, const std::string &type = TexType.diffuse,
          bool gamma = false, bool flipTexture = false,
          bool autoFreeData_ = true)
      : type(type), autoFreeData(autoFreeData_) {
    load(path, gamma, flipTexture);
  }
  ~Texture() { freeHostData(); }
  unsigned int id;
  std::string type;
  std::string path;
  unsigned char *data;
  bool autoFreeData;

  unsigned int load(const std::string &path, bool gamma = false,
                    bool flipTexture = false) {
    stbi_set_flip_vertically_on_load(flipTexture);
    unsigned int textureID;
    glGenTextures(1, &textureID);
    int width, height, nrComponents;
    data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
    if (data) {
      GLenum format;
      if (nrComponents == 1)
        format = GL_RED;
      else if (nrComponents == 3)
        format = GL_RGB;
      else if (nrComponents == 4)
        format = GL_RGBA;

      glBindTexture(GL_TEXTURE_2D, textureID);
      glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format,
                   GL_UNSIGNED_BYTE, data);
      glGenerateMipmap(GL_TEXTURE_2D);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                      GL_LINEAR_MIPMAP_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      // stbi_image_free(data);
    } else {
      DYM_WARNING_cs("Texture", std::string("Texture failed to load at path: " +
                                            std::string(path))
                                    .c_str());
      // stbi_image_free(data);
    }
    if (autoFreeData) freeHostData();
    stbi_set_flip_vertically_on_load(false);
    id = textureID;
    this->path = path;
    return textureID;
  }
  void freeHostData() {
    if (!data) return;
    stbi_image_free(data);
    data = nullptr;
  }
};
}  // namespace rdt
}  // namespace dym
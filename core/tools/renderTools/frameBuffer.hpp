#pragma once

// gl,pic
#include "../../dyGraphic.hpp"

// math
#include "../../dyMath.hpp"
#include "math/define.hpp"
#include "tools/renderTools/modelLoader.hpp"
#include <string>

namespace dym {
namespace rdt {
class FrameBuffer {
public:
  uReali fbo;
  uReali texture, rbo;
  std::string name;
  FrameBuffer(const uReali &SCR_WIDTH = 800, const uReali &SCR_HEIGHT = 800,
              const std::string &name = "FrameBuffer")
      : name(name) {
    // gen Framebuffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      DYM_ERROR_cs(name, "Failed to bind frameBuffer.");

    // gen textureBuffer
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           texture, 0);

    // gen rbo
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH,
                          SCR_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, rbo);

    close();
  }
  ~FrameBuffer() { glDeleteFramebuffers(1, &fbo); }

  void use(Vector4 clearColor = {0, 0, 0, 0}) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
  }
  void close() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
};
} // namespace rdt
} // namespace dym
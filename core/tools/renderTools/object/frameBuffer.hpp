#pragma once

// gl,pic
#include "./object.hpp"

// math
#include "dyMath.hpp"
#include "tools/renderTools/modelLoader.hpp"
#include <string>

namespace dym {
namespace rdo {
template <bool useDraw = false> class FrameBuffer : public renderObject {
public:
  uReali fbo;
  uReali texture, rbo;
  unsigned int VBO, VAO;
  FrameBuffer(const std::string &name = "FrameBuffer",
              const uReali &SCR_WIDTH = 800, const uReali &SCR_HEIGHT = 800)
      : renderObject(name) {
    // gen Framebuffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

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
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH,
                          SCR_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, rbo);

    close();
    if constexpr (useDraw) {
      float quadVertices[] = {-1.0f, 1.0f, 0.0f, 1.0f,  -1.0f, -1.0f,
                              0.0f,  0.0f, 1.0f, -1.0f, 1.0f,  0.0f,

                              -1.0f, 1.0f, 0.0f, 1.0f,  1.0f,  -1.0f,
                              1.0f,  0.0f, 1.0f, 1.0f,  1.0f,  1.0f};
      glGenVertexArrays(1, &VAO);
      glGenBuffers(1, &VBO);
      glBindVertexArray(VAO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices,
                   GL_STATIC_DRAW);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                            (void *)0);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                            (void *)(2 * sizeof(float)));
    }
  }
  ~FrameBuffer() { glDeleteFramebuffers(1, &fbo); }

  void use(Vector4 clearColor = {0, 0, 0, 0}) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glEnable(GL_DEPTH_TEST);
  }
  void close() {
    // check framebuffer
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      DYM_ERROR_cs("FrameBuffer", "Framebuffer is not complete!");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  // layout(location=0)in vec2 Pos;
  // layout(location=1)in vec2 TexCoords;
  virtual void Draw(rdt::Shader &shader, unsigned int instancedNum = 1) {
    glBindVertexArray(VAO);
    if (instancedNum > 1)
      glDrawArraysInstanced(GL_TRIANGLES, 0, 6, instancedNum);
    else
      glDrawArrays(GL_TRIANGLES, 0, 6);
  }
};
} // namespace rdo
} // namespace dym
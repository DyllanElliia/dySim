#pragma once

// gl,pic
#include "./object.hpp"

// math
#include "dyMath.hpp"
#include "tools/renderTools/modelLoader.hpp"
#include <string>

namespace dym {
namespace rdo {
template <bool useDraw = false> class GBuffer : public renderObject {
public:
  uReali gbuffer;
  uReali gPosition, gNormal, gAlbedoSpec, rboDepth;
  unsigned int VBO, VAO;
  GBuffer(const std::string &name = "GBuffer", const uReali &SCR_WIDTH = 800,
          const uReali &SCR_HEIGHT = 800)
      : renderObject(name) {
    // gen Framebuffer
    glGenFramebuffers(1, &gbuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, gbuffer);

    // - Position color buffer
    glGenTextures(1, &gPosition);
    glBindTexture(GL_TEXTURE_2D, gPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB,
                 GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           gPosition, 0);
    // - Normal color buffer
    glGenTextures(1, &gNormal);
    glBindTexture(GL_TEXTURE_2D, gNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB,
                 GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                           gNormal, 0);
    // - Color + Specular color buffer
    glGenTextures(1, &gAlbedoSpec);
    glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D,
                           gAlbedoSpec, 0);
    // - Tell OpenGL which color attachments we'll use (of this framebuffer) for
    // rendering
    GLuint attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                             GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, attachments);

    // gen rbo
    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH,
                          SCR_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rboDepth);

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
  ~GBuffer() { glDeleteFramebuffers(1, &gbuffer); }

  void use(Vector4 clearColor = {0, 0, 0, 0}) {
    glBindFramebuffer(GL_FRAMEBUFFER, gbuffer);
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glEnable(GL_DEPTH_TEST);
  }
  void close() {
    // check framebuffer
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      DYM_ERROR_cs(name, "FrameBuffer is not complete!");
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
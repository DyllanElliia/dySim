/*
 * @Author: DyllanElliia
 * @Date: 2021-11-12 16:02:04
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2021-11-13 18:10:48
 * @Description:
 */
#pragma once

#include <glad/glad.h>

#include <GLFW/glfw3.h>

// #include <GL/glut.h>

#include "src/tensor.hpp"

#include <glm/glm.hpp>

#include <learnopengl/camera.h>
#include <learnopengl/shader.h>

// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}

namespace dym {
enum ViewMode { VIEWER_2D, VIEWER_3D };

class GUI {
 private:
  ViewMode viewMode;
  std::vector<std::function<void(GLFWwindow *)>> processInput;
  std::vector<unsigned int> VAO, VBO_v, VBO_c;  // VAO -> VBO : 1 -> 2
  unsigned int VxO_i;

  inline bool checkObjId(unsigned int &i) {
    unsigned int VAO_size = VAO.size();
    if (i > VAO_size) {
      qp_ctrl(tType::UNDERLINE, tColor::YELLOW);
      qprint("GUI::checkObjId WORNING: VAO Index is too large!");
      qp_ctrl();
      i = VAO_size;
      // return false;
    }
    if (i == VAO_size) {
      VAO.push_back(0);
      VBO_v.push_back(0);
      VBO_c.push_back(0);
      glGenVertexArrays(1, &VAO[i]);
      glGenBuffers(1, &VBO_v[i]);
      glGenBuffers(1, &VBO_c[i]);
    }
    return true;
  }

 public:
  GLFWwindow *window;
  std::string windowName;
  Index<float> background_color;
  std::vector<Shader> shaderList;
  Camera camera();

  GUI(std::string windowName_ = "dyMath", ViewMode viewMode_ = VIEWER_2D,
      Index<int> background_color_ = Index<int>(3, 0))
      : window(nullptr),
        windowName(windowName_),
        viewMode(viewMode_),
        VxO_i(0) {
    for (auto &bc : background_color_) background_color.push_back(bc / 255.0);
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    processInput.push_back([](GLFWwindow *window) {
      if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    });
    shaderList.push_back(
        Shader("../shader/default.vs", "../shader/default.frag"));
  }
  ~GUI() { glfwTerminate(); }

  bool init(const unsigned int src_width, const unsigned int src_height) {
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
      qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::RED);
      qprint("Failed to create GLFW window");
      qp_ctrl();
      glfwTerminate();
      return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::RED);
      qprint("Failed to initialize GLAD");
      qp_ctrl();
      return false;
    }
  }

  bool scatter2D(Tensor<float> &loc, Index<int> color_default,
                 unsigned int begin = 0, unsigned int end = -1) {
    Index locShape = loc.shape();
    auto &vec_num = locShape[0];
    auto &vec_d = locShape[1];
    if (vec_d != 2) {
      qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::RED);
      qprint(
          "GUI::scatter2D ERROW: vertex dimension is no equal to 2!\nSTOP "
          "DRAWING!");
      qp_ctrl();
      return false;
    }
    if (end == -1) end = vec_num;
    Index<float> color;
    for (auto &c : color_default) color.push_back(c / 255.0);
    checkObjId(VxO_i);
    glBindVertexArray(VAO[VxO_i]);
    // add vertex
    glBindBuffer(GL_ARRAY_BUFFER, VBO_v[VxO_i]);
    glBufferData(GL_ARRAY_BUFFER, (end - begin) * vec_d * sizeof(float),
                 &loc[begin * vec_d], GL_STREAM_DRAW);
    glVertexAttribPointer(0, vec_d, GL_FLOAT, GL_FALSE, vec_d * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);
    // add color
    glBindBuffer(GL_ARRAY_BUFFER, VBO_c[VxO_i]);
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float), color.a, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    ++VxO_i;
    return true;
  }

  bool update(std::function<void()> updateFun) { return true; }
};
}  // namespace dym
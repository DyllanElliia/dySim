/*
 * @Author: DyllanElliia
 * @Date: 2021-11-12 16:02:04
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-05-23 15:47:54
 * @Description:
 */
#pragma once

// glad
#include <glad/glad.h>
// glad must include before the glfw library.
#include <GLFW/glfw3.h>

// #include <GL/glut.h>

#include <learnopengl/camera.h>
#include <learnopengl/shader.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#include "math/tensor.hpp"
#include "math/vector.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#endif

bool *keys;
GLfloat lastX, lastY;
double m_xoffset = 0, m_yoffset = 0, s_yoffset = 0;
bool firstMouse;
namespace dym {
enum ViewMode { VIEWER_2D, VIEWER_3D };

#ifndef _dym_pixel_typedef_
#define _dym_pixel_typedef_
typedef unsigned char Pixel;
#endif
class GUI {
 private:
  ViewMode viewMode;
  std::vector<std::function<void(GLFWwindow *)>> processInput;
  std::vector<unsigned int> VAO, VBO_v, VBO_c, u_shader,
      EBO;  // VAO -> VBO : 1 -> 2
  std::vector<glm::vec3> color_l;
  std::vector<std::pair<unsigned short, unsigned int>> draw_property;
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
      // create AND gen
      VAO.push_back(0);
      VBO_v.push_back(0);
      // VBO_c.push_back(0);
      EBO.push_back(0);
      color_l.push_back(glm::vec3(0));
      draw_property.push_back(std::make_pair(0, 0));
      u_shader.push_back(0);
      glGenVertexArrays(1, &VAO[i]);
      glGenBuffers(1, &VBO_v[i]);
      // glGenBuffers(1, &VBO_c[i]);
      glGenBuffers(1, &EBO[i]);
    }
    return true;
  }

  // CallBack_function
  bool run = true;
  // glfw: whenever the window size changed (by OS or user resize) this callback
  // function executes
  // ---------------------------------------------------------------------------------------------
  static void framebuffer_size_callback(GLFWwindow *window, int width,
                                        int height) {
    // make sure the viewport matches the new window dimensions; note that width
    // and height will be significantly larger than specified on retina
    // displays.
    glViewport(0, 0, width, height);
  }
  // Is called whenever a key is pressed/released via GLFW
  static void key_callback(GLFWwindow *window, int key, int scancode,
                           int action, int mode) {
    // cout << key << endl;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
      glfwSetWindowShouldClose(window, GL_TRUE);
    if (key >= 0 && key < 1024) {
      if (action == GLFW_PRESS)
        keys[key] = true;
      else if (action == GLFW_RELEASE)
        keys[key] = false;
      qprint("GUI message: reveive keyboard call key =", key, (char)key);
    }
  }

  static void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    if (firstMouse) {
      lastX = xpos;
      lastY = ypos;
      firstMouse = false;
    }

    GLfloat xoffset = xpos - lastX;
    GLfloat yoffset =
        lastY - ypos;  // Reversed since y-coordinates go from bottom to left

    lastX = xpos;
    lastY = ypos;
    m_xoffset += xoffset, m_yoffset += yoffset;
    // qprint("GUI message: reveive mouse call pos = (", xoffset, ",", yoffset,
    //        ")");
    // camera.ProcessMouseMovement(xoffset, yoffset);
  }

  static void scroll_callback(GLFWwindow *window, double xoffset,
                              double yoffset) {
    s_yoffset += yoffset;
    // camera.ProcessMouseScroll(yoffset);
  }

 public:
  GLFWwindow *window;
  std::string windowName;
  Index<float> background_color;
  std::vector<Shader> shaderList;
  Camera camera;
  unsigned int src_height, src_width;

  GUI(std::string windowName_ = "dyMath",
      Index<int> background_color_ = Index<int>(3, 0),
      ViewMode viewMode_ = VIEWER_2D)
      : window(nullptr),
        windowName(windowName_),
        viewMode(viewMode_),
        VxO_i(0) {
    keys = new bool[1024];
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
  }
  ~GUI() {
    for (auto &v : VAO) glDeleteVertexArrays(1, &v);
    // for (auto &v : VBO_c)
    //   glDeleteBuffers(1, &v);
    for (auto &v : VBO_v) glDeleteBuffers(1, &v);
    for (auto &v : EBO) glDeleteBuffers(1, &v);
    glfwTerminate();
    delete[] keys;
    qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::GREEN);
    qprint("dySim GUI end!");
    qp_ctrl();
  }

  bool init(const unsigned int src_width_, const unsigned int src_height_) {
    lastX = src_width_ >> 1, lastY = src_height_ >> 1, firstMouse = true;
    src_width = src_width_, src_height = src_height_;
    window =
        glfwCreateWindow(src_width, src_height, windowName.c_str(), NULL, NULL);
    if (window == NULL) {
      qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::RED);
      qprint("Failed to create GLFW window");
      qp_ctrl();
      glfwTerminate();
      return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::RED);
      qprint("Failed to initialize GLAD");
      qp_ctrl();
      return false;
    }

    // add Input call_back
    processInput.push_back([](GLFWwindow *window) {
      if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    });
    // add default shader
    switch (viewMode) {
      case VIEWER_2D:
        shaderList.push_back(
            Shader("../shader/default2D.vs", "../shader/default2D.frag"));
        shaderList.push_back(
            Shader("../shader/default2Dpic.vs", "../shader/default2Dpic.frag"));
        break;
      case VIEWER_3D:
        shaderList.push_back(
            Shader("../shader/default3D.vs", "../shader/default3D.frag"));
        shaderList.push_back(
            Shader("../shader/default2Dpic.vs", "../shader/default2Dpic.frag"));
        break;
      default:
        qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::RED);
        qprint("GUI ERROR: Init default shader failure!");
        qp_ctrl();
        return false;
    }
    // qprint("fin");

    int sw, sh;
    glfwGetFramebufferSize(window, &sw, &sh);
    glViewport(0, 0, sw, sh);

    if (viewMode == VIEWER_3D)
      // Setup some OpenGL options
      glEnable(GL_DEPTH_TEST);

    qprint("-------------------------------------------------");
    qprint("************* Welcome to use dySim! *************");
    qprint("-------------------------------------------------");
    qprint("OpenGL  version:  4.6");
    qprint("dySim   version:  0.5 (Unreleased)");
    qprint("Author: DyllanElliia");
    qprint("Github: https://github.com/DyllanElliia/dySim\n");

    return true;
  }

  bool scatter2D(Tensor<Vector<Real, 2>> &loc, Index<int> color_default,
                 int shader_index = 0, unsigned int begin = 0,
                 unsigned int end = -1) {
    Index locShape = loc.shape();
    auto &vec_num = locShape[0];
    if (end > vec_num) end = vec_num;
    checkObjId(VxO_i);
    u_shader[VxO_i] = shader_index;
    // add color
    auto &color = color_l[VxO_i];
    color[0] = color_default[0] / 255.0;
    color[1] = color_default[1] / 255.0;
    color[2] = color_default[2] / 255.0;
    glBindVertexArray(VAO[VxO_i]);
    // add vertex
    glBindBuffer(GL_ARRAY_BUFFER, VBO_v[VxO_i]);
    glBufferData(GL_ARRAY_BUFFER, (end - begin) * sizeof(Vector<Real, 2>),
                 &loc[begin], GL_STREAM_DRAW);
    glVertexAttribPointer(0, 2, GL_DOUBLE, GL_FALSE, sizeof(Vector<Real, 2>),
                          (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    draw_property[VxO_i] = std::make_pair(GL_POINTS, end - begin);

    ++VxO_i;
    return true;
  }

  template <std::size_t color_size>
  bool imshow(Tensor<Vector<Pixel, color_size>> &pic, int shader_index = 1) {
    Index picShape = pic.shape();
    if (picShape.size() != 2) return false;
    auto y = picShape[0], x = picShape[1];
    checkObjId(VxO_i);
    u_shader[VxO_i] = shader_index;
    glEnable(GL_TEXTURE_2D);
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions        // colors         // texture coords
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f   // top left
    };
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };
    // unsigned int VBO, VAO, EBO;
    glBindVertexArray(VAO[VxO_i]);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_v[VxO_i]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[VxO_i]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // load and create a texture
    // -------------------------
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D,
                  texture);  // all upcoming GL_TEXTURE_2D operations now have
                             // effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    GL_REPEAT);  // set texture wrapping to GL_REPEAT (default
                                 // wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, x, y, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 &(pic[0][0]));
    // glGenerateMipmap(GL_TEXTURE_2D);

    draw_property[VxO_i] = std::make_pair(texture, 0);

    ++VxO_i;
    // delete[] data;
    return true;
  }

  // template <std::size_t color_size>
  // _DYM_FORCE_INLINE_ bool imshow(const Tensor<Vector<Real, color_size>> &pic,
  //                                int shader_index = 1) {
  //   qprint("here1");
  //   Tensor<Vector<Pixel, color_size>> picPix(0, pic.shape());
  //   picPix.for_each_i(
  //       [&](Vector<Pixel, PIC_RGB> &e, int i) { e = pic[i].cast<Pixel>(); });
  //   qprint(pic[1000], picPix[1000], pi(pic.shape()), pi(picPix.shape()));
  //   qprint("here2");
  //   imshow(picPix, shader_index);
  // }

  bool update(std::function<void()> updateFun) {
    qp_ctrl(tType::BOLD, tType::UNDERLINE, tColor::GREEN);
    qprint("dySim GUI run!");
    qp_ctrl();
    while (!glfwWindowShouldClose(window)) {
      for (auto &fun : processInput) fun(window);
      auto save_i = VxO_i;
      // update all
      updateFun();

      // draw src

      glfwPollEvents();
      // Clear the colorbuffer
      glClearColor(background_color[0], background_color[1],
                   background_color[2], 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      // use default shader

      if (viewMode == VIEWER_2D) {
        for (int i = 0; i < VxO_i; ++i) {
          auto &u_shader_i = u_shader[i];
          auto &ourShader = shaderList[u_shader_i];
          switch (u_shader_i) {
            case 0: {
              ourShader.use();
              ourShader.setVec3("color", color_l[i]);
              glBindVertexArray(VAO[i]);
              glPointSize(3);
              glDrawArrays(draw_property[i].first, 0, draw_property[i].second);
              break;
            }
            case 1: {
              // bind Texture
              glActiveTexture(GL_TEXTURE0);
              glBindTexture(GL_TEXTURE_2D, draw_property[i].first);

              // render container
              ourShader.use();
              ourShader.setInt("ourTexture", 0);
              glBindVertexArray(VAO[i]);
              glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

              break;
            }
            default: {
              qp_ctrl(tColor::YELLOW, tType::BOLD, tType::UNDERLINE);
              qprint("dyGUI Warning: Unknown shader index (", u_shader_i,
                     ", in:", i, ").");
              qp_ctrl();
              getchar();
              break;
            }
          }
        }
        glBindVertexArray(0);
      } else if (viewMode == VIEWER_3D) {
        int deltaTime = 1e-3;
        auto do_move = [&]() {
          camera.ProcessMouseMovement(m_xoffset, m_yoffset);
          camera.ProcessMouseScroll(s_yoffset);
          m_xoffset = 0, m_yoffset = 0, s_yoffset = 0;

          // Camera controls
          if (keys[GLFW_KEY_W]) camera.ProcessKeyboard(FORWARD, deltaTime);
          if (keys[GLFW_KEY_S]) camera.ProcessKeyboard(BACKWARD, deltaTime);
          if (keys[GLFW_KEY_A]) camera.ProcessKeyboard(LEFT, deltaTime);
          if (keys[GLFW_KEY_D]) camera.ProcessKeyboard(RIGHT, deltaTime);
          if (keys[GLFW_KEY_SPACE]) run = false;
        };
        // Create camera transformation
        glm::mat4 view;
        view = camera.GetViewMatrix();
        glm::mat4 projection;
        projection = glm::perspective(
            camera.Zoom, (float)src_width / (float)src_height, 0.1f, 1000.0f);
        glm::mat4 model = glm::translate(model, glm::vec3(0, 0, 0));

        for (int i = 0; i < VxO_i; ++i) {
          auto &ourShader = shaderList[u_shader[i]];
          // Get the uniform locations
          ourShader.setMat4("view", view);
          ourShader.setMat4("projection", projection);
          ourShader.setMat4("model", model);
          glBindVertexArray(VAO[i]);
          glPointSize(2);
          glDrawArrays(draw_property[i].first, 0, draw_property[i].second);
        }
        glBindVertexArray(0);
      }

      VxO_i = save_i;
      // Swap the buffers
      glfwSwapBuffers(window);
    }
    return true;
  }
};
}  // namespace dym
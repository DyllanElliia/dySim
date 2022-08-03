/*
 * @Author: DyllanElliia
 * @Date: 2022-01-23 19:33:53
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-09 15:10:57
 * @Description:
 */

// #include "learnopengl/shader.h"
#include "tools/renderTools/modelLoader.hpp"
#include <dyGraphic.hpp>
#include <learnopengl/camera.h>
// #include <learnopengl/model.h>
#include <learnopengl/shader_m.h>

#include <iostream>

int main() {
  const unsigned int SCR_WIDTH = 1000;
  const unsigned int SCR_HEIGHT = 1000;
  dym::GUI gui("asdf", dym::gi(10, 50, 10), dym::ViewMode::VIEWER_3D);
  gui.init(SCR_WIDTH, SCR_HEIGHT);
  Shader ourShader("./shader/model_loading.vs", "./shader/model_loading.fs");

  dym::Model ourModel("./assets/nanosuit/nanosuit.obj");

  Camera &camera = gui.camera;
  camera.Position = {0, 0, 3};
  gui.update(
      [&]() {
        ourShader.use();
        glm::mat4 projection = glm::perspective(
            glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT,
            0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        // render the loaded model
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));

        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);
        ourShader.setMat4("model", model);
        ourModel.Draw(ourShader);
      },
      0.001);

  return 0;
}

// void framebuffer_size_callback(GLFWwindow *window, int width, int height);
// void mouse_callback(GLFWwindow *window, double xpos, double ypos);
// void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
// void processInput(GLFWwindow *window, Camera &camera);

// // settings
// const unsigned int SCR_WIDTH = 800;
// const unsigned int SCR_HEIGHT = 600;

// // camera

// float lastX = SCR_WIDTH / 2.0f;
// float lastY = SCR_HEIGHT / 2.0f;
// bool firstMouse = true;

// // timing
// float deltaTime = 0.0f;
// float lastFrame = 0.0f;

// int main() {
//   const unsigned int SCR_WIDTH = 1000;
//   const unsigned int SCR_HEIGHT = 1000;
//   dym::GUI gui("asdf", dym::gi(10, 50, 10), dym::ViewMode::VIEWER_3D);
//   gui.init(SCR_WIDTH, SCR_HEIGHT);

//   auto &window = gui.window;

//   qprint("here");
//   // build and compile shaders
//   // -------------------------
//   Shader ourShader("./shader/model_loading.vs", "./shader/model_loading.fs");

//   dym::Model ourModel("./assets/nanosuit/nanosuit.obj");

//   // draw in wireframe
//   // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

//   Camera &camera = gui.camera;

//   // render loop
//   // -----------
//   gui.update(
//       [&]() {
//         // don't forget to enable shader before setting uniforms
//         ourShader.use();

//         // view/projection transformations
//         glm::mat4 projection = glm::perspective(
//             glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT,
//             0.1f, 100.0f);
//         glm::mat4 view = camera.GetViewMatrix();
//         ourShader.setMat4("projection", projection);
//         ourShader.setMat4("view", view);

//         // render the loaded model
//         glm::mat4 model = glm::mat4(1.0f);
//         model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
//         model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));
//         ourShader.setMat4("model", model);
//         ourModel.Draw(ourShader);
//       },
//       0.001);
//   return 0;
// }

// void processInput(GLFWwindow *window, Camera &camera) {
//   // if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
//   //   glfwSetWindowShouldClose(window, true);

//   if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
//     camera.ProcessKeyboard(FORWARD, deltaTime);
//   if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
//     camera.ProcessKeyboard(BACKWARD, deltaTime);
//   if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
//     camera.ProcessKeyboard(LEFT, deltaTime);
//   if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
//     camera.ProcessKeyboard(RIGHT, deltaTime);
// }

// void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
//   glViewport(0, 0, width, height);
// }

// void mouse_callback(GLFWwindow *window, double xposIn, double yposIn) {
//   float xpos = static_cast<float>(xposIn);
//   float ypos = static_cast<float>(yposIn);

//   if (firstMouse) {
//     lastX = xpos;
//     lastY = ypos;
//     firstMouse = false;
//   }

//   float xoffset = xpos - lastX;
//   float yoffset =
//       lastY - ypos; // reversed since y-coordinates go from bottom to top

//   lastX = xpos;
//   lastY = ypos;

//   // camera.ProcessMouseMovement(xoffset, yoffset);
// }

// void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
//   // camera.ProcessMouseScroll(static_cast<float>(yoffset));
// }
/*
 * @Author: DyllanElliia
 * @Date: 2022-01-23 19:33:53
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-09 15:10:57
 * @Description:
 */

#include <dyGraphic.hpp>

#include <iostream>

int main() {
  const unsigned int SCR_WIDTH = 1000;
  const unsigned int SCR_HEIGHT = 1000;
  dym::GUI gui("asdf", dym::gi(10, 50, 10), dym::ViewMode::VIEWER_3D);
  gui.init(SCR_WIDTH, SCR_HEIGHT);
  dym::Shader ourShader("./shader/model_loading.vs",
                        "./shader/model_loading.fs");

  dym::Model ourModel("./assets/nanosuit/nanosuit.obj");

  dym::Camera &camera = gui.camera;
  camera.Position = {0, 5, 3};
  gui.update(
      [&]() {
        ourShader.use();
        dym::Matrix<lReal, 4, 4> projection = glm::perspective(
            glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT,
            0.1f, 100.0f);
        dym::Matrix<lReal, 4, 4> view = camera.GetViewMatrix();
        // render the loaded model
        dym::Matrix<lReal, 4, 4> model = glm::mat4(1.0f);
        model = glm::translate(model.to_glm_mat(), {0.0f, 0.0f, 0.0f});
        model = glm::scale(model.to_glm_mat(), {1.0f, 1.0f, 1.0f});

        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);
        ourShader.setMat4("model", model);
        ourShader.setVec3("offsets[0]", {-5, 0, 0});
        ourShader.setVec3("offsets[0]", {5, 0, 0});
        ourModel.Draw(ourShader, 2);
      },
      0.001);

  return 0;
}

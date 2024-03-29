/*
 * @Author: DyllanElliia
 * @Date: 2022-02-28 16:37:43
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-02-28 16:41:54
 * @Description:
 */
#include <dyGraphic.hpp>

#include <iostream>

int main() {
  const unsigned int SCR_WIDTH = 1000;
  const unsigned int SCR_HEIGHT = 1000;
  dym::GUI gui("asdf", dym::gi(0, 0, 0), dym::ViewMode::VIEWER_3D);
  gui.init(SCR_WIDTH, SCR_HEIGHT);
  dym::rdt::Shader ourShader("./shader/gbuffer.vs", "./shader/gbuffer.fs");
  dym::rdt::Shader lightShader("./shader/lightbox.vs", "./shader/lightbox.fs");
  dym::rdt::Shader screenShader("./shader/screenbf.vs", "./shader/screenbf.fs");

  dym::rdt::Model ourModel("./assets/nanosuit_reflection/nanosuit.obj");
  dym::rdo::Cube lightCube;

  dym::rdo::GBuffer<true> gbuffer;

  dym::rdt::Camera &camera = gui.camera;
  camera.Position = {0, 10, 20};
  // camera.Position = {0, 0, 0};
  dym::Vector3 lightPos{0, 0, 0}, lightColor(1.0);

  auto setCameraMatrix = [&](dym::rdt::Shader &s, dym::Matrix4l &p,
                             dym::Matrix4l &v, dym::Matrix4l &m) {
    s.setMat4("projection", p);
    s.setMat4("view", v);
    s.setMat4("model", m);
  };
  int i = 0;
  gui.update(
      [&]() {
        dym::Matrix4l projection = glm::perspective(
            glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT,
            0.1f, 100.0f);
        dym::Matrix4l view = camera.GetViewMatrix();
        dym::Matrix4l model = glm::mat4(1.0f);
        model = glm::translate(model.to_glm_mat(), {0.0f, 0.0f, 0.0f});
        model = glm::scale(model.to_glm_mat(), {1.0f, 1.0f, 1.0f});

        gbuffer.use();

        auto setOurShader = [&](dym::rdt::Shader &s) {
          s.use();
          setCameraMatrix(s, projection, view, model);
          s.setVec3("offsets[0]", {-4, 0, 0});
          s.setVec3("offsets[1]", {4, 0, 0});
          s.setVec3("viewPos", camera.Position);
        };
        setOurShader(ourShader);
        ourModel.Draw(ourShader, 2);

        gbuffer.close();

        screenShader.use();
        screenShader.setTexture("gNormal", gbuffer.gNormal);
        screenShader.setTexture("gPos", gbuffer.gPosition);
        screenShader.setTexture("gApSp", gbuffer.gAlbedoSpec);
        gbuffer.Draw(screenShader);
      },
      0.001);

  return 0;
}

// int main() {
//   const unsigned int SCR_WIDTH = 1000;
//   const unsigned int SCR_HEIGHT = 1000;
//   dym::GUI gui("asdf", dym::gi(0, 0, 0), dym::ViewMode::VIEWER_3D);
//   gui.init(SCR_WIDTH, SCR_HEIGHT);
//   dym::rdt::Shader ourShader("./shader/model_loading.vs",
//                              "./shader/model_loading3tex.fs");
//   dym::rdt::Shader lightShader("./shader/lightbox.vs",
//   "./shader/lightbox.fs"); dym::rdt::Shader
//   screenShader("./shader/screenbf.vs", "./shader/screenbf.fs");

//   dym::rdt::Model ourModel("./assets/nanosuit/nanosuit.obj");
//   dym::rdo::Cube lightCube;
//   dym::rdt::Material mat({1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {0.5, 0.5, 0.5},
//                          32.);
//   dym::rdt::PoLightMaterial lmat({1.0, 1.0, 1.0}, {1.0, 1.0, 1.0},
//                                  {1.0, 1.0, 1.0}, 1.0, 0.045, 0.0075);

//   dym::rdo::FrameBuffer<true> framebuf;

//   dym::rdt::Camera &camera = gui.camera;
//   camera.Position = {4, 10, 20};
//   dym::Vector3 lightPos{4, 0, 0}, lightColor(1.0);

//   auto setCameraMatrix = [&](dym::rdt::Shader &s, dym::Matrix4l &p,
//                              dym::Matrix4l &v, dym::Matrix4l &m) {
//     s.setMat4("projection", p);
//     s.setMat4("view", v);
//     s.setMat4("model", m);
//   };
//   int i = 0;
//   gui.update(
//       [&]() {
//         lightPos[1] = 10 + 10 * dym::sin((i) / 30.0);
//         lightPos[2] = 0. + 10 * dym::cos((i++) / 30.0);
//         lmat.position = lightPos;

//         dym::Matrix4l projection = glm::perspective(
//             glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT,
//             0.1f, 100.0f);
//         dym::Matrix4l view = camera.GetViewMatrix();
//         dym::Matrix4l model = glm::mat4(1.0f);
//         model = glm::translate(model.to_glm_mat(), {0.0f, 0.0f, 0.0f});
//         model = glm::scale(model.to_glm_mat(), {1.0f, 1.0f, 1.0f});

//         framebuf.use();

//         ourShader.use();
//         setCameraMatrix(ourShader, projection, view, model);
//         ourShader.setVec3("offsets[0]", {-8, 0, 0});
//         ourShader.setVec3("offsets[0]", {8, 0, 0});
//         ourShader.setVec3("viewPos", camera.Position);
//         ourShader.setMaterial("material", mat);
//         ourShader.setLightMaterial("light", lmat);
//         ourModel.Draw(ourShader, 2);

//         lightShader.use();
//         setCameraMatrix(lightShader, projection, view, model);
//         lightShader.setVec3("lightPos", lightPos);
//         lightShader.setVec3("lightColor", lightColor);
//         lightCube.Draw(lightShader);

//         framebuf.close();

//         screenShader.use();
//         screenShader.setTexture("screenTexture", 0, framebuf.texture);
//         framebuf.Draw(screenShader);
//       },
//       0.001);

//   return 0;
// }
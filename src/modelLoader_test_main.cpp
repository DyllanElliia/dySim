/*
 * @Author: DyllanElliia
 * @Date: 2022-04-12 16:21:12
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-12 17:39:03
 * @Description:
 */
#include "../core/tools/modelLoader.hpp"

int main(int argc, char const *argv[]) {
  dym::Model bunny("./assets/bunny.ply");
  qprint(bunny.meshes.size(), bunny.textures_loaded.size(),
         bunny.meshes[0].faces.size(), bunny.meshes[0].faces[0]);
  return 0;
}

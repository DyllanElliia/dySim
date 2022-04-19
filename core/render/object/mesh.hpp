/*
 * @Author: DyllanElliia
 * @Date: 2022-04-11 14:22:27
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-18 18:13:20
 * @Description:
 */
#pragma once

#include "../BVH/bvhNode.hpp"
#include "triangle.hpp"

// dym model loader
#include "../../tools/modelLoader.hpp"

namespace dym {
namespace rt {

class Mesh : public Hittable {
 private:
  int createMesh(std::vector<Vector3ui>& faces, shared_ptr<Material>& m) {
    HittableList world;
    world.objects.resize(faces.size());
#pragma omp parallel for
    for (int i = 0; i < faces.size(); ++i)
      world.objects[i] = std::make_shared<Triangle>(vertices[faces[i][0]],
                                                    vertices[faces[i][1]],
                                                    vertices[faces[i][2]], m);

    worlds = make_shared<BvhNode>(world);
    return faces.size();
  }

 public:
  Mesh() {}
  Mesh(std::vector<Point3>& positions, std::vector<Vector3>& normals,
       std::vector<Vector3ui>& faces, shared_ptr<Material> m);
  Mesh(std::vector<Point3>& positions, std::vector<Vector3ui>& faces,
       shared_ptr<Material> m);
  Mesh(std::vector<Vertex>& vertices_, std::vector<Vector3ui>& faces,
       shared_ptr<Material> m);
  Mesh(dym::Mesh& mesh_, shared_ptr<Material> default_mat);

  void reBuild(dym::Mesh& mesh_, shared_ptr<Material> default_mat);

  virtual bool hit(const Ray& r, Real t_min, Real t_max,
                   HitRecord& rec) const override;
  virtual bool bounding_box(aabb& output_box) const override;

  virtual Real pdf_value(const Point3& origin, const Vector3& v) const override;
  virtual Vector3 random(const Point3& origin) const override;

 public:
  std::vector<Vertex> vertices;
  shared_ptr<BvhNode> worlds;
};

Mesh::Mesh(std::vector<Point3>& positions, std::vector<Vector3>& normals,
           std::vector<Vector3ui>& faces, shared_ptr<Material> m) {
  if (positions.size() != normals.size())
    DYM_ERROR(
        "DYM::RT::MESH ERROR: Positions's size must be equal to Normals's "
        "size");
  for (int i = 0; i < positions.size(); ++i)
    vertices.push_back(Vertex(positions[i], normals[i], 0, 1));
  createMesh(faces, m);
}

Mesh::Mesh(std::vector<Point3>& positions, std::vector<Vector3ui>& faces,
           shared_ptr<Material> m) {
  std::vector<Vector3> normals(positions.size());
  std::vector<unsigned short> normals_n(positions.size(), 0);
#pragma omp parallel for
  for (int i = 0; i < faces.size(); ++i) {
    auto& index = faces[i];
    Point3 &v0 = positions[index[0]], &v1 = positions[index[1]],
           &v2 = positions[index[2]];
    //  TODO: compute normal
    auto normal = (v1 - v0).cross(v2 - v0);
#pragma omp critical
    Loop<int, 3>([&](auto j) {
      auto& indexj = index[j];
      normals[indexj] += normal, normals_n[indexj]++;
    });
  }
#pragma omp parallel for
  for (int i = 0; i < normals.size(); ++i)
    normals[i] = (normals[i] / Real(normals_n[i])).normalize();
  for (int i = 0; i < positions.size(); ++i)
    vertices.push_back(Vertex(positions[i], normals[i], 0, 1));
  createMesh(faces, m);
}

Mesh::Mesh(std::vector<Vertex>& vertices_, std::vector<Vector3ui>& faces,
           shared_ptr<Material> m) {
  vertices = vertices_;
  createMesh(faces, m);
}

Mesh::Mesh(dym::Mesh& mesh_, shared_ptr<Material> default_mat) {
  vertices.resize(mesh_.vertices.size());
  auto& mesh_vertices = mesh_.vertices;
#pragma omp parallel for
  for (int i = 0; i < vertices.size(); ++i) {
    vertices[i] =
        Vertex(mesh_vertices[i].Position, mesh_vertices[i].Normal,
               mesh_vertices[i].TexCoords[0], mesh_vertices[i].TexCoords[1]);
  }
  createMesh(mesh_.faces, default_mat);
}

void Mesh::reBuild(dym::Mesh& mesh_, shared_ptr<Material> default_mat) {
  vertices.resize(mesh_.vertices.size());
  auto& mesh_vertices = mesh_.vertices;
#pragma omp parallel for
  for (int i = 0; i < vertices.size(); ++i) {
    vertices[i] =
        Vertex(mesh_vertices[i].Position, mesh_vertices[i].Normal,
               mesh_vertices[i].TexCoords[0], mesh_vertices[i].TexCoords[1]);
  }
  createMesh(mesh_.faces, default_mat);
}

bool Mesh::hit(const Ray& r, Real t_min, Real t_max, HitRecord& rec) const {
  if (worlds->hit(r, t_min, t_max, rec)) {
    rec.obj_id = (int)(std::size_t)this;
    return true;
  } else
    return false;
}

bool Mesh::bounding_box(aabb& output_box) const {
  return worlds->bounding_box(output_box);
}

Real Mesh::pdf_value(const Point3& origin, const Vector3& v) const {
  DYM_ERROR(
      "DYM::RT::MESH ERROR: Developers have not provided the pdf_value "
      "function "
      "yet");
  return 0;
}
Vector3 Mesh::random(const Point3& origin) const {
  DYM_ERROR(
      "DYM::RT::MESH ERROR: Developers have not provided the random "
      "function "
      "yet");
  return 0;
}

}  // namespace rt
}  // namespace dym
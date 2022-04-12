/*
 * @Author: DyllanElliia
 * @Date: 2022-04-12 15:49:45
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-12 17:38:28
 * @Description:
 */

#pragma once
// gl,pic
#include "../dyGraphic.hpp"

// math
#include "../dyMath.hpp"

// assimp
// importer
#include <assimp/Importer.hpp>
// process
#include <assimp/postprocess.h>
// scene
#include <assimp/scene.h>

namespace dym {

typedef Vector<Real, 2> Vector2;
typedef Vector<unsigned int, 3> Vector3ui;
#define MAX_BONE_INFLUENCE 4

struct Vertex {
  // position
  Vector3 Position;
  // normal
  Vector3 Normal;
  // texCoords
  Vector2 TexCoords;
  // tangent
  Vector3 Tangent;
  // bitangent
  Vector3 Bitangent;
  // bone indexes which will influence this vertex
  int m_BoneIDs[MAX_BONE_INFLUENCE];
  // weights from each bone
  Real m_Weights[MAX_BONE_INFLUENCE];
};

struct Texture {
  unsigned int id;
  std::string type;
  std::string path;
};

class Mesh {
 public:
  // mesh Data
  std::vector<Vertex> vertices;
  std::vector<Vector3ui> faces;
  std::vector<Texture> textures;
  unsigned int VAO;

  // constructor
  Mesh(std::vector<Vertex> vertices, std::vector<Vector3ui> faces,
       std::vector<Texture> textures) {
    qprint("here?");
    this->vertices = vertices;
    this->faces = faces;
    this->textures = textures;
    qprint("or here?");

    // now that we have all the required data, set the vertex buffers and its
    // attribute pointers.
    // setupMesh();
    qprint("other?");
  }

  // render the mesh
  void Draw(Shader &shader) {
    // bind appropriate textures
    unsigned int diffuseNr = 1;
    unsigned int specularNr = 1;
    unsigned int normalNr = 1;
    unsigned int heightNr = 1;
    for (unsigned int i = 0; i < textures.size(); i++) {
      glActiveTexture(GL_TEXTURE0 +
                      i);  // active proper texture unit before binding
      // retrieve texture number (the N in diffuse_textureN)
      std::string number;
      std::string name = textures[i].type;
      if (name == "texture_diffuse")
        number = std::to_string(diffuseNr++);
      else if (name == "texture_specular")
        number =
            std::to_string(specularNr++);  // transfer unsigned int to stream
      else if (name == "texture_normal")
        number = std::to_string(normalNr++);  // transfer unsigned int to stream
      else if (name == "texture_height")
        number = std::to_string(heightNr++);  // transfer unsigned int to stream

      // now set the sampler to the correct texture unit
      glUniform1i(glGetUniformLocation(shader.ID, (name + number).c_str()), i);
      // and finally bind the texture
      glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }

    // draw mesh
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, faces.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // always good practice to set everything back to defaults once configured.
    glActiveTexture(GL_TEXTURE0);
  }

 private:
  // render data
  unsigned int VBO, EBO;

  // initializes all the buffer objects/arrays
  void setupMesh() {
    // create buffers/arrays
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // A great thing about structs is that their memory layout is sequential for
    // all its items. The effect is that we can simply pass a pointer to the
    // struct and it translates perfectly to a Vector3/2 array which again
    // translates to 3/2 floats which translates to a byte array.
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex),
                 &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(unsigned int),
                 &faces[0], GL_STATIC_DRAW);

    // set the vertex attribute pointers
    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(Vertex), (void *)0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, Normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_DOUBLE, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, TexCoords));
    // vertex tangent
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_DOUBLE, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, Tangent));
    // vertex bitangent
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_DOUBLE, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, Bitangent));
    // ids
    glEnableVertexAttribArray(5);
    glVertexAttribIPointer(5, 4, GL_INT, sizeof(Vertex),
                           (void *)offsetof(Vertex, m_BoneIDs));

    // weights
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 4, GL_DOUBLE, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, m_Weights));
    glBindVertexArray(0);
  }
};
namespace {
unsigned int TextureFromFile(const char *path, const std::string &directory,
                             bool gamma = false) {
  std::string filename = std::string(path);
  filename = directory + '/' + filename;

  unsigned int textureID;
  glGenTextures(1, &textureID);
  int width, height, nrComponents;
  unsigned char *data =
      stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
  if (data) {
    GLenum format;
    if (nrComponents == 1)
      format = GL_RED;
    else if (nrComponents == 3)
      format = GL_RGB;
    else if (nrComponents == 4)
      format = GL_RGBA;

    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format,
                 GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);
  } else {
    DYM_WARNING(
        std::string("Texture failed to load at path: " + std::string(path))
            .c_str());
    stbi_image_free(data);
  }

  return textureID;
}
}  // namespace

class Model {
 public:
  // model data
  std::vector<Texture>
      textures_loaded;  // stores all the textures loaded so far, optimization
                        // to make sure textures aren't loaded more than once.
  std::vector<Mesh> meshes;
  std::string directory;
  bool gammaCorrection;

  // constructor, expects a filepath to a 3D model.
  Model(std::string const &path, bool gamma = false) : gammaCorrection(gamma) {
    loadModel(path);
  }

  // draws the model, and thus all its meshes
  void Draw(Shader &shader) {
    for (unsigned int i = 0; i < meshes.size(); i++) meshes[i].Draw(shader);
  }

 private:
  // loads a model with supported ASSIMP extensions from file and stores the
  // resulting meshes in the meshes vector.
  void loadModel(std::string const &path) {
    // read file via ASSIMP
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(
        path, aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                  aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
    // check for errors
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
        !scene->mRootNode)  // if is Not Zero
    {
      DYM_ERROR(
          std::string("ASSIMP ERROR: " + std::string(importer.GetErrorString()))
              .c_str());
      return;
    }
    // retrieve the directory path of the filepath
    directory = path.substr(0, path.find_last_of('/'));

    // process ASSIMP's root node recursively
    processNode(scene->mRootNode, scene);
  }

  // processes a node in a recursive fashion. Processes each individual mesh
  // located at the node and repeats this process on its children nodes (if
  // any).
  void processNode(aiNode *node, const aiScene *scene) {
    qprint("in processNode");
    // process each mesh located at the current node
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
      // the node object only contains faces to index the actual objects in
      // the scene. the scene contains all the data, node is just to keep stuff
      // organized (like relations between nodes).
      aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
      meshes.push_back(processMesh(mesh, scene));
    }
    // after we've processed all of the meshes (if any) we then recursively
    // process each of the children nodes
    for (unsigned int i = 0; i < node->mNumChildren; i++) {
      processNode(node->mChildren[i], scene);
    }
  }

  Mesh processMesh(aiMesh *mesh, const aiScene *scene) {
    qprint("in processMesh", mesh->mNumVertices);
    // data to fill
    std::vector<Vertex> vertices;
    std::vector<Vector3ui> faces;
    std::vector<Texture> textures;

    // walk through each of the mesh's vertices
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
      // qprint(i);
      Vertex vertex;
      Vector3 vector;  // we declare a placeholder vector since assimp uses
                       // its own vector class that doesn't directly convert
                       // to Vector3 class so we transfer the data to this
                       // placeholder Vector3 first.
      // positions
      vector[0] = mesh->mVertices[i].x;
      vector[1] = mesh->mVertices[i].y;
      vector[2] = mesh->mVertices[i].z;
      vertex.Position = vector;
      if (i == 0) qprint(vector);
      if (i == 35945) qprint(vector);
      if (i == 35946) qprint(vector);

      if (i == 35946) qprint("normal");
      // normals
      if (mesh->HasNormals()) {
        vector[0] = mesh->mNormals[i].x;
        vector[1] = mesh->mNormals[i].y;
        vector[2] = mesh->mNormals[i].z;
        vertex.Normal = vector;
      }

      if (i == 35946) qprint("texture");

      // texture coordinates
      if (mesh->mTextureCoords[0])  // does the mesh contain texture
                                    // coordinates?
      {
        Vector2 vec;
        // a vertex can contain up to 8 different texture coordinates. We thus
        // make the assumption that we won't use models where a vertex can have
        // multiple texture coordinates so we always take the first set (0).
        vec[0] = mesh->mTextureCoords[0][i].x;
        vec[1] = mesh->mTextureCoords[0][i].y;
        vertex.TexCoords = vec;
        // tangent
        vector[0] = mesh->mTangents[i].x;
        vector[1] = mesh->mTangents[i].y;
        vector[2] = mesh->mTangents[i].z;
        vertex.Tangent = vector;
        // bitangent
        vector[0] = mesh->mBitangents[i].x;
        vector[1] = mesh->mBitangents[i].y;
        vector[2] = mesh->mBitangents[i].z;
        vertex.Bitangent = vector;
      } else
        vertex.TexCoords = Vector2({0.0f, 0.0f});

      if (i == 35946) qprint("end");

      vertices.push_back(vertex);
    }
    qprint("next", mesh->mNumFaces);
    // now wak through each of the mesh's faces (a face is a mesh its triangle)
    // and retrieve the corresponding vertex faces.
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
      aiFace face = mesh->mFaces[i];
      // retrieve all faces of the face and store them in the faces vector
      // for (unsigned int j = 0; j < face.mNumIndices; j++)
      //   faces.push_back(face.mIndices[j]);
      faces.push_back(
          Vector3ui({face.mIndices[0], face.mIndices[1], face.mIndices[2]}));
    }
    // process materials
    aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
    // we assume a convention for sampler names in the shaders. Each diffuse
    // texture should be named as 'texture_diffuseN' where N is a sequential
    // number ranging from 1 to MAX_SAMPLER_NUMBER. Same applies to other
    // texture as the following list summarizes: diffuse: texture_diffuseN
    // specular: texture_specularN
    // normal: texture_normalN

    // 1. diffuse maps
    std::vector<Texture> diffuseMaps = loadMaterialTextures(
        material, aiTextureType_DIFFUSE, "texture_diffuse");
    textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
    // 2. specular maps
    std::vector<Texture> specularMaps = loadMaterialTextures(
        material, aiTextureType_SPECULAR, "texture_specular");
    textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
    // 3. normal maps
    std::vector<Texture> normalMaps =
        loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal");
    textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
    // 4. height maps
    std::vector<Texture> heightMaps =
        loadMaterialTextures(material, aiTextureType_AMBIENT, "texture_height");
    textures.insert(textures.end(), heightMaps.begin(), heightMaps.end());

    // return a mesh object created from the extracted mesh data
    return Mesh(vertices, faces, textures);
  }

  // checks all material textures of a given type and loads the textures if
  // they're not loaded yet. the required info is returned as a Texture struct.
  std::vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type,
                                            std::string typeName) {
    std::vector<Texture> textures;
    qprint("loadMater", mat->GetTextureCount(type));
    for (unsigned int i = 0; i < mat->GetTextureCount(type); i++) {
      aiString str;
      mat->GetTexture(type, i, &str);
      // check if texture was loaded before and if so, continue to next
      // iteration: skip loading a new texture
      bool skip = false;
      for (unsigned int j = 0; j < textures_loaded.size(); j++) {
        if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0) {
          textures.push_back(textures_loaded[j]);
          skip = true;  // a texture with the same filepath has already been
                        // loaded, continue to next one. (optimization)
          break;
        }
      }
      if (!skip) {  // if texture hasn't been loaded already, load it
        Texture texture;
        texture.id = TextureFromFile(str.C_Str(), this->directory);
        texture.type = typeName;
        texture.path = str.C_Str();
        textures.push_back(texture);
        textures_loaded.push_back(
            texture);  // store it as texture loaded for entire model, to ensure
                       // we won't unnecesery load duplicate textures.
      }
    }
    return textures;
  }
};

}  // namespace dym
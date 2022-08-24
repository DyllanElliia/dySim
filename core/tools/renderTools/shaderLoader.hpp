#pragma once

// gl,pic
#include "../../dyGraphic.hpp"

// math
#include "../../dyMath.hpp"
#include "math/define.hpp"

// basic
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// textureRegister
#include "./textureRegister.hpp"

namespace dym {
namespace rdt {
struct BaseMaterial {};

struct Material : public BaseMaterial {
  Vector3l ambient;
  Vector3l diffuse;
  Vector3l specular;
  lReal shininess;
  Material(const Vector3l &ambient, const Vector3l &diffuse,
           const Vector3l &specular, const lReal shininess)
      : ambient(ambient), diffuse(diffuse), specular(specular),
        shininess(shininess) {}
};

struct LightMaterial : public BaseMaterial {
  Vector3 position;
  Vector3l ambient;
  Vector3l diffuse;
  Vector3l specular;

  LightMaterial(const Vector3l &ambient, const Vector3l &diffuse,
                const Vector3l &specular)
      : ambient(ambient), diffuse(diffuse), specular(specular) {}
};

struct PaLightMaterial : public BaseMaterial {
  Vector3 direction;
  Vector3l ambient;
  Vector3l diffuse;
  Vector3l specular;

  PaLightMaterial(const Vector3l &ambient, const Vector3l &diffuse,
                  const Vector3l &specular)
      : ambient(ambient), diffuse(diffuse), specular(specular) {}
};

struct PoLightMaterial : public BaseMaterial {
  Vector3 position;
  Vector3l ambient;
  Vector3l diffuse;
  Vector3l specular;

  lReal constant;
  lReal linear;
  lReal quadratic;
  PoLightMaterial(const Vector3l &ambient, const Vector3l &diffuse,
                  const Vector3l &specular, const lReal &constant = 1,
                  const lReal &linear = 0.9, const lReal &quadratic = 0.032)
      : ambient(ambient), diffuse(diffuse), specular(specular),
        constant(constant), linear(linear), quadratic(quadratic) {}
};

struct SpotLightMaterial : public BaseMaterial {
  Vector3 position;
  Vector3 direction;
  lReal inCutOff, outCutOff;

  Vector3l ambient;
  Vector3l diffuse;
  Vector3l specular;

  SpotLightMaterial(const Vector3l &ambient, const Vector3l &diffuse,
                    const Vector3l &specular, const lReal &inCutOff,
                    const lReal &outCutOff)
      : ambient(ambient), diffuse(diffuse), specular(specular),
        inCutOff(inCutOff), outCutOff(outCutOff) {}
};

class Shader {
public:
  unsigned int ID;
  std::string name="Shader";
  // constructor generates the shader on the fly
  // ------------------------------------------------------------------------
  Shader(const char *vertexPath, const char *fragmentPath,
         const char *geometryPath = nullptr) {
    // 1. retrieve the vertex/fragment source code from filePath
    std::string vertexCode;
    std::string fragmentCode;
    std::string geometryCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;
    std::ifstream gShaderFile;
    // ensure ifstream objects can throw exceptions:
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      // open files
      vShaderFile.open(vertexPath);
      fShaderFile.open(fragmentPath);
      std::stringstream vShaderStream, fShaderStream;
      // read file's buffer contents into streams
      vShaderStream << vShaderFile.rdbuf();
      fShaderStream << fShaderFile.rdbuf();
      // close file handlers
      vShaderFile.close();
      fShaderFile.close();
      // convert stream into string
      vertexCode = vShaderStream.str();
      fragmentCode = fShaderStream.str();
      // if geometry shader path is present, also load a geometry shader
      if (geometryPath != nullptr) {
        gShaderFile.open(geometryPath);
        std::stringstream gShaderStream;
        gShaderStream << gShaderFile.rdbuf();
        gShaderFile.close();
        geometryCode = gShaderStream.str();
      }
    } catch (std::ifstream::failure &e) {
      std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
    }
    const char *vShaderCode = vertexCode.c_str();
    const char *fShaderCode = fragmentCode.c_str();
    // 2. compile shaders
    // fragment Shader
    auto fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");
    // vertex shader
    auto vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");

    // if geometry shader is given, compile geometry shader
    unsigned int geometry;
    if (geometryPath != nullptr) {
      const char *gShaderCode = geometryCode.c_str();
      geometry = glCreateShader(GL_GEOMETRY_SHADER);
      glShaderSource(geometry, 1, &gShaderCode, NULL);
      glCompileShader(geometry);
      checkCompileErrors(geometry, "GEOMETRY");
    }
    // shader Program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    if (geometryPath != nullptr)
      glAttachShader(ID, geometry);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");
    // delete the shaders as they're linked into our program now and no longer
    // necessery
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (geometryPath != nullptr)
      glDeleteShader(geometry);
  }
  // activate the shader
  // ------------------------------------------------------------------------
  void use() { glUseProgram(ID); }
  // utility uniform functions
  // ------------------------------------------------------------------------
  void setBool(const std::string &name, bool value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
  }
  // ------------------------------------------------------------------------
  void setInt(const std::string &name, int value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
  }
  // ------------------------------------------------------------------------
  void setFloat(const std::string &name, float value) const {
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
  }
  // ------------------------------------------------------------------------
  void setTexture(const std::string &name, int index,
                  unsigned int textureID) const {
    glActiveTexture(GL_TEXTURE0 + index);
    glUniform1i(glGetUniformLocation(ID, name.c_str()), index);
    glBindTexture(GL_TEXTURE_2D, textureID);
  }
  void setTexture(const std::string &name, unsigned int textureID) const {
    auto index = texRegFind(ID, name);
    glActiveTexture(GL_TEXTURE0 + index);
    glUniform1i(glGetUniformLocation(ID, name.c_str()), index);
    glBindTexture(GL_TEXTURE_2D, textureID);
  }
  // ------------------------------------------------------------------------
  void setVec2(const std::string &name, glm::vec2 value) const {
    glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
  }
  void setVec2(const std::string &name, float x, float y) const {
    glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
  }
  // ------------------------------------------------------------------------
  void setVec3(const std::string &name, glm::vec3 value) const {
    glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
  }
  void setVec3(const std::string &name, float x, float y, float z) const {
    glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
  }
  // ------------------------------------------------------------------------
  void setVec4(const std::string &name, glm::vec4 value) const {
    glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
  }
  void setVec4(const std::string &name, float x, float y, float z, float w) {
    glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
  }
  // ------------------------------------------------------------------------
  void setMat2(const std::string &name, glm::mat2 mat) const {
    glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE,
                       &mat[0][0]);
  }
  // ------------------------------------------------------------------------
  void setMat3(const std::string &name, glm::mat3 mat) const {
    glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE,
                       &mat[0][0]);
  }
  // ------------------------------------------------------------------------
  void setMat4(const std::string &name, glm::mat4 mat) const {
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE,
                       &mat[0][0]);
  }
  // ------------------------------------------------------------------------
  void setMaterial(const std::string &name, const Material &mat) {
    setVec3(name + ".ambient", mat.ambient);
    setVec3(name + ".diffuse", mat.diffuse);
    setVec3(name + ".specular", mat.specular);
    setFloat(name + ".shininess", mat.shininess);
  }
  void setLightMaterial(const std::string &name, const LightMaterial &mat) {
    setVec3(name + ".position", mat.position);
    setVec3(name + ".ambient", mat.ambient);
    setVec3(name + ".diffuse", mat.diffuse);
    setVec3(name + ".specular", mat.specular);
  }
  void setLightMaterial(const std::string &name, const PaLightMaterial &mat) {
    setVec3(name + ".direction", mat.direction);
    setVec3(name + ".ambient", mat.ambient);
    setVec3(name + ".diffuse", mat.diffuse);
    setVec3(name + ".specular", mat.specular);
  }
  void setLightMaterial(const std::string &name, const PoLightMaterial &mat) {
    setVec3(name + ".position", mat.position);
    setVec3(name + ".ambient", mat.ambient);
    setVec3(name + ".diffuse", mat.diffuse);
    setVec3(name + ".specular", mat.specular);
    setFloat(name + ".constant", mat.constant);
    setFloat(name + ".linear", mat.linear);
    setFloat(name + ".quadratic", mat.quadratic);
  }
  void setLightMaterial(const std::string &name, const SpotLightMaterial &mat) {
    setVec3(name + ".position", mat.position);
    setVec3(name + ".direction", mat.direction);
    setFloat(name + ".inCutOff", mat.inCutOff);
    setFloat(name + ".outCutOff", mat.outCutOff);
    setVec3(name + ".ambient", mat.ambient);
    setVec3(name + ".diffuse", mat.diffuse);
    setVec3(name + ".specular", mat.specular);
  }

private:
  // utility function for checking shader compilation/linking errors.
  // ------------------------------------------------------------------------
  void checkCompileErrors(GLuint shader, std::string type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
      glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
      if (!success) {
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        DYM_ERROR_cs(name,
                     "SHADER_COMPILATION_ERROR of type: " + type + "\n" +
                         std::string(infoLog));
      }
    } else {
      glGetProgramiv(shader, GL_LINK_STATUS, &success);
      if (!success) {
        glGetProgramInfoLog(shader, 1024, NULL, infoLog);
        DYM_ERROR_cs(name,
                     "PROGRAM_LINKING_ERROR of type: " + type + "\n" +
                         std::string(infoLog));
      }
    }
  }
};
} // namespace rdt
} // namespace dym
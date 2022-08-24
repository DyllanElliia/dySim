#version 330 core
layout(location=0)in vec3 aPos;
layout(location=1)in vec3 aNormal;
layout(location=2)in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;

layout(std140)uniform Object
{
  mat4 projection;
  mat4 view;
};

uniform mat4 model;
// uniform mat4 view;
// uniform mat4 projection;

uniform vec3 offsets[2];

void main()
{
  vec3 offset=offsets[gl_InstanceID];
  TexCoords=aTexCoords;
  Normal=aNormal;
  FragPos=vec3(model*vec4(aPos+offset,1.));
  gl_Position=projection*view*model*vec4(aPos+offset,1.);
}
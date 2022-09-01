#version 330 core
layout(location=0)in vec3 aPos;
layout(location=1)in vec3 aNormal;
layout(location=2)in vec2 aTexCoords;
layout(location=3)in vec3 tangent;
layout(location=4)in vec3 bitangent;

out vec2 TexCoords;
// out vec3 Normal;
out vec3 FragPos;
out mat3 TBN;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
// uniform mat4 view;
// uniform mat4 projection;

uniform vec3 offsets[2];

void main()
{
  vec3 offset=offsets[gl_InstanceID];
  TexCoords=aTexCoords;
  // Normal=aNormal;
  FragPos=vec3(model*vec4(aPos+offset,1.));
  gl_Position=projection*view*model*vec4(aPos+offset,1.);
  vec3 T=normalize(vec3(model*vec4(tangent,0.)));
  vec3 B=normalize(vec3(model*vec4(bitangent,0.)));
  vec3 N=normalize(vec3(model*vec4(aNormal,0.)));
  TBN=mat3(T,B,N);
}
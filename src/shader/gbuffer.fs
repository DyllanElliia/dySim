#version 330 core

layout(location=0)out vec3 gPosition;
layout(location=1)out vec3 gNormal;
layout(location=2)out vec4 gAlbedoSpec;

in vec2 TexCoords;
in vec3 FragPos;
// in vec3 Normal;
in mat3 TBN;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;
uniform sampler2D texture_normal1;
uniform sampler2D texture_height1;

void main()
{
  gPosition=FragPos;
  gNormal=normalize(TBN*normalize(texture(texture_normal1,TexCoords).rgb*2.-1.));
  gAlbedoSpec.rgb=texture(texture_diffuse1,TexCoords).rgb;
  gAlbedoSpec.a=texture(texture_specular1,TexCoords).r;
}
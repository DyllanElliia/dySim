#version 330 core

layout(location=0)out vec4 gPosition;
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

const float NEAR=.01;
const float FAR=100.f;
float LinearizeDepth(float depth)
{
  float z=depth*2.-1.;// -> NDC
  return(2.*NEAR*FAR)/(FAR+NEAR-z*(FAR-NEAR));
}

void main()
{
  gPosition=vec4(FragPos,(gl_FragCoord.z));
  gNormal=normalize(TBN*normalize(texture(texture_normal1,TexCoords).rgb*2.-1.));
  gAlbedoSpec.rgb=texture(texture_diffuse1,TexCoords).rgb;
  gAlbedoSpec.a=texture(texture_specular1,TexCoords).r;
}
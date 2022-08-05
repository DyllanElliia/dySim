#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D texture_diffuse1;

// light
uniform vec3 lightPos;
uniform vec3 lightColor;

// Color
uniform vec3 ambient;

void main()
{
  vec3 objColor=vec3(texture(texture_diffuse1,TexCoords));
  
  vec3 norm=normalize(Normal);
  vec3 lightDir=normalize(lightPos-FragPos);
  float diff=max(dot(norm,lightDir),0.);
  vec3 diffuse=diff*lightColor;
  
  vec3 result=(ambient+diffuse)*objColor;
  FragColor=vec4(result,1.);
}
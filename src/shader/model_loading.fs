#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;
uniform sampler2D texture_normal1;
// uniform sampler2D texture_height1;

// light
struct Light{
  vec3 position;
  
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
  
  float constant;
  float linear;
  float quadratic;
};

uniform Light light;

// view
uniform vec3 viewPos;

// Color
// uniform vec3 ambient;

// material
struct Material{
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
  float shininess;
};

uniform Material material;

uniform samplerCube skybox;

void main()
{
  vec3 objColor=vec3(texture(texture_diffuse1,TexCoords));
  vec3 objSpec=vec3(texture(texture_specular1,TexCoords));
  vec3 objNorm=vec3(texture(texture_normal1,TexCoords));
  vec3 ambient=light.ambient*material.ambient;
  vec3 norm=normalize(Normal);
  vec3 lightPos=light.position;
  vec3 lightDir=normalize(lightPos-FragPos);
  
  // diffuse
  float diff=max(dot(norm,lightDir),0.);
  vec3 diffuse=light.diffuse*(diff*material.diffuse);
  
  // specular
  vec3 viewDir=normalize(viewPos-FragPos);
  vec3 reflectDir=reflect(-lightDir,norm);
  float spec=pow(max(dot(viewDir,reflectDir),0.),material.shininess);
  vec3 specular=objSpec*light.specular*(spec*material.specular);
  
  // attenuation
  float distance=length(lightPos-FragPos);
  float attenuation=1./(light.constant+light.linear*distance+
    light.quadratic*(distance*distance));
    
    // SkyBox
    vec3 R=reflect(-viewDir,norm);
    vec3 sky=vec3(texture(skybox,R));
    // vec3 sky=vec3(1.,1.,1.);
    
    vec3 result=(ambient+diffuse+specular)*objColor*attenuation+sky;
    FragColor=vec4(result,1.);
  }
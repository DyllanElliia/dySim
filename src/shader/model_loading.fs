#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
// in vec3 Normal;
in vec3 FragPos;
in mat3 TBN;

// layout(std140,binding=0)uniform Object{samplerCube skybox;};

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;
uniform sampler2D texture_normal1;
uniform sampler2D texture_height1;

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
  vec3 objheig=vec3(texture(texture_height1,TexCoords));
  vec3 ambient=light.ambient*material.ambient;
  vec3 norm=normalize(TBN*normalize(objNorm*2.-1.));
  vec3 lightPos=light.position;
  vec3 lightDir=normalize(lightPos-FragPos);
  
  // diffuse
  float diff=max(dot(norm,lightDir),0.);
  vec3 diffuse=light.diffuse*(diff*material.diffuse);
  
  // specular
  vec3 viewDir=normalize(viewPos-FragPos);
  vec3 halfwayDir=normalize(lightDir+viewDir);
  // vec3 reflectDir=reflect(-lightDir,norm);
  // float spec=pow(max(dot(viewDir,reflectDir),0.),material.shininess);
  float spec=pow(max(dot(norm,halfwayDir),0.),material.shininess);
  vec3 specular=objSpec*light.specular*(spec*material.specular);
  
  // attenuation
  float distance=length(lightPos-FragPos);
  float attenuation=1./(light.constant+light.linear*distance+
    light.quadratic*(distance*distance));
    
    // SkyBox
    vec3 R=reflect(-viewDir,norm);
    vec3 sky=vec3(texture(skybox,R))*max(objheig,.05);
    // vec3 sky=vec3(1.,1.,1.);
    
    vec3 result=(ambient+diffuse+specular)*objColor*attenuation+sky;
    // vec3 result=objheig;
    
    FragColor=vec4(result,1.);
    // gamma
    float gamma=1.2;
    FragColor.rgb=pow(FragColor.rgb,vec3(1./gamma));
  }
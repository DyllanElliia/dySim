#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;

// void main()
// {
  //   vec3 col=texture(screenTexture,TexCoords).rgb;
  //   FragColor=vec4(col,1.);
// }

const float offset=1./300.;

void main()
{
  vec2 offsets[9]=vec2[](
    vec2(-offset,offset),// 左上
    vec2(0.f,offset),// 正上
    vec2(offset,offset),// 右上
    vec2(-offset,0.f),// 左
    vec2(0.f,0.f),// 中
    vec2(offset,0.f),// 右
    vec2(-offset,-offset),// 左下
    vec2(0.f,-offset),// 正下
    vec2(offset,-offset)// 右下
  );
  
  float kernel[9]=float[](
    -1,-1,-1,
    -1,9,-1,
    -1,-1,-1
  );
  
  vec3 sampleTex[9];
  for(int i=0;i<9;i++)
  {
    sampleTex[i]=vec3(texture(screenTexture,TexCoords.st+offsets[i]));
  }
  vec3 col=vec3(0.);
  for(int i=0;i<9;i++)
  col+=sampleTex[i]*kernel[i];
  
  FragColor=vec4(col,1.);
}
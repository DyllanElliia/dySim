#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gNormal;
uniform sampler2D gPos;
uniform sampler2D gApSp;

void main()
{
  // FragColor=vec4(texture(gNormal,TexCoords).rgb,1.);
  vec2 tc;
  vec4 result;
  if(TexCoords.x>.5){
    tc.x=TexCoords.x*2-1;
    if(TexCoords.y>.5){
      tc.y=TexCoords.y*2-1;
      result=vec4(texture(gNormal,tc).rgb,1.);
    }else{
      tc.y=TexCoords.y*2;
      result=vec4(vec3(texture(gApSp,tc).a),1.);
    }
  }else{
    tc.x=TexCoords.x*2;
    if(TexCoords.y>.5){
      tc.y=TexCoords.y*2-1;
      // result=vec4(texture(gPos,tc).rgb,1.);
      result=vec4(vec3(pow(texture(gPos,tc).a,50.)),1.);
    }else{
      tc.y=TexCoords.y*2;
      result=vec4(texture(gApSp,tc).rgb,1.);
    }
  }
  FragColor=result;
}
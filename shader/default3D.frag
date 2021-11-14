#version 330 core
in vec3 vertexColor;

out vec4 color;

uniform sampler2D ourTexture1;
uniform sampler2D ourTexture2;

void main()
{
    color = vec4(vertexColor, 1.0);
}
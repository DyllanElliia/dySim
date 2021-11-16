#version 330 core
layout (location = 0) in vec2 position;
// layout (location = 1) in vec3 vertexColor_;

uniform vec3 color;

out vec3 vertexColor;


void main()
{
    gl_Position = vec4(vec3(position, 0.0), 1.0);
    vertexColor = color;
}
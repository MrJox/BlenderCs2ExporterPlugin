#version 330 core

in vec2 fragTexCoord;
out vec4 FragColor;

uniform sampler2D textureSampler;

void main()
{
    vec4 texColor = texture(textureSampler, fragTexCoord);
    FragColor = texColor;
}

#ifndef GLSHADER_H
#define GLSHADER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

GLuint LoadShader(const char* vertex_path, const char* fragment_path);

#endif
#pragma once
#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>

void error_callback(int error, const char* description);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
unsigned int create_shader_program(const char* vertexShaderSource, const char* fragmentShaderSource);
#pragma once
#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "GLHelpers.h"
#include "Window.h"


unsigned int create_shader_program(const char* vertexShader, const char* fragmentShader);
void mainloop(GLFWwindow* window);

void error_callback(int error, const char* description);
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
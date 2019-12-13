#pragma once

#ifndef _RENDERING_HEADER_
#define _RENDERING_HEADER_

#if KEVIN_MACOS
#pragma message("Included on Mac OS")
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "stb_image.h"
#include <librealsense2/rs.hpp>
#include <stdio.h>
#include <stdlib.h>

//#include <GLFW/glfw3.h>

#define GL_GLEXT_PROTOTYPES

//#include <GL/GL.h>

#include <librealsense2/rs.hpp> 

#include <string>
#include <sstream>
#include <iostream>

#include "shader.hpp"

namespace vc::rendering {
    const float COLOR_vertices[] = {
         0.0f, 1.0f,  1.0f, 1.0f, // top right
        0.0f, 0.f,  1.0f, 0.0f, // bottom right
        -1.f, 0.f,  0.0f, 0.0f, // bottom left
        -1.f, 1.f,  0.0f, 1.0f  // top left 
    };
    const unsigned int COLOR_indices[] = {  // note that we start from 0!
        0, 1, 3,   // first triangle
        1, 2, 3    // second triangle
    };

    class Rendering {
    private:
        unsigned int COLOR_VBO, COLOR_VAO, COLOR_EBO, COLOR_texture;
        vc::rendering::Shader* COLOR_shader;

    public:
        Rendering() {
            COLOR_shader = new vc::rendering::Shader("shader/only_color.vs", "shader/only_color.fs");

            glGenVertexArrays(1, &COLOR_VAO);
            glGenBuffers(1, &COLOR_VBO);
            glGenBuffers(1, &COLOR_EBO);

            glBindVertexArray(COLOR_VAO);

            glBindBuffer(GL_ARRAY_BUFFER, COLOR_VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(COLOR_vertices), COLOR_vertices, GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, COLOR_EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(COLOR_indices), COLOR_indices, GL_STATIC_DRAW);

            // position attribute
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            // color attribute
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
            glEnableVertexAttribArray(1);
             
            glGenTextures(1, &COLOR_texture);
            glBindTexture(GL_TEXTURE_2D, COLOR_texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
            // set the texture wrapping parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            // set texture filtering parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        void renderOnlyColor(rs2::frame color_image, const float pos_x, const float pos_y, const float aspect) {
            const int width = color_image.as<rs2::video_frame>().get_width();
            const int height = color_image.as<rs2::video_frame>().get_height();
            glBindTexture(GL_TEXTURE_2D, COLOR_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, color_image.get_data());

            glBindTexture(GL_TEXTURE_2D, COLOR_texture);

            const float image_aspect = 1.0f * width / height;

            float x_aspect = 1;
            float y_aspect = 1;
            if (image_aspect < aspect) {
                x_aspect = image_aspect / aspect;
            }
            else {
                y_aspect = aspect / image_aspect;
            }

            COLOR_shader->use();
            COLOR_shader->setVec2("offset", pos_x, pos_y);
            COLOR_shader->setVec2("aspect", x_aspect, y_aspect );
            glBindVertexArray(COLOR_VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);

            std::cout << std::endl;
           /* glBindBuffer(GL_ARRAY_BUFFER, COLOR_VBO);
            glBufferData(GL_ARRAY_BUFFER, vertices.size(), vertices.data(), GL_STREAM_DRAW);*/
        }
    };

    void startFrame(GLFWwindow* window) {
        glfwMakeContextCurrent(window);
        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
}

#endif // !_RENDERING_HEADER_

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
    void setViewport(const int viewport_width, const int viewport_height, const int pos_x, const int pos_y);

    glm::mat4 COORDINATE_CORRECTION = glm::mat4(
        -1.0f, 0,0,0,
        0, -1.0f, 0,0,
        0,0, 1.0f, 0,
        0,0,0, 1.0f
    );

    const glm::vec3 DEBUG_COLORS[4] = {
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(1.0f, 1.0f, 1.0f)
    };

    const float COLOR_vertices[] = {
         1.0f,  1.0f,  1.0f, 0.0f, // top right
         1.0f, -1.0f,  1.0f, 1.0f, // bottom right
        -1.0f, -1.0f,  0.0f, 1.0f, // bottom left
        -1.0f,  1.0f,  0.0f, 0.0f  // top left 
    };
    const unsigned int COLOR_indices[] = {  // note that we start from 0!
        0, 1, 3,   // first triangle
        1, 2, 3    // second triangle
    };

    const float COORDINATE_SYSTEM[] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.01f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.01f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };

    class Rendering {
    private:
        unsigned int COORDINATE_SYSTEM_VBO, COORDINATE_SYSTEM_VAO;
        unsigned int VBOs[3], VAOs[2], EBOs[1], textures[2];
        vc::rendering::Shader* TEXTURE_shader;
        vc::rendering::Shader* POINTCLOUD_shader;
        vc::rendering::Shader* COORDINATE_SYSTEM_shader;

    public:
        Rendering() {
            TEXTURE_shader = new vc::rendering::VertexFragmentShader("shader/texture.vs", "shader/texture.fs");
            POINTCLOUD_shader = new vc::rendering::VertexFragmentShader("shader/pointcloud.vs", "shader/pointcloud.fs");
            COORDINATE_SYSTEM_shader = new vc::rendering::VertexFragmentShader("shader/coordinate.vs", "shader/coordinate.fs");

            glGenVertexArrays(2, VAOs);
            glGenVertexArrays(1, &COORDINATE_SYSTEM_VAO);
            glGenBuffers(3, VBOs);
            glGenBuffers(1, &COORDINATE_SYSTEM_VBO);
            glGenBuffers(1, EBOs);
            glGenTextures(2, textures);

            initializeVerticesBuffer();
            initializeTextureBuffer();
            initializeCoordinateSystemBuffers();
        }

        void initializeCoordinateSystemBuffers() {
            glBindVertexArray(COORDINATE_SYSTEM_VAO);
            glBindBuffer(GL_ARRAY_BUFFER, COORDINATE_SYSTEM_VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(COORDINATE_SYSTEM), COORDINATE_SYSTEM, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
        }

        void initializeTextureBuffer() {
            glBindVertexArray(VAOs[0]);

            glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(COLOR_vertices), COLOR_vertices, GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOs[0]);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(COLOR_indices), COLOR_indices, GL_STATIC_DRAW);

            // position attribute
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            // color attribute
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
            glEnableVertexAttribArray(1);

            glBindTexture(GL_TEXTURE_2D, textures[0]); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
            // set the texture wrapping parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            // set texture filtering parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        void initializeVerticesBuffer() {
            glBindVertexArray(VAOs[1]);
            glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
            glBindBuffer(GL_ARRAY_BUFFER, VBOs[2]);

            glBindTexture(GL_TEXTURE_2D, textures[1]); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
            // set the texture wrapping parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            // set texture filtering parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        void renderAllPointclouds(const rs2::points points, const rs2::frame texture, glm::mat4 model, glm::mat4 view, glm::mat4 projection,
            const int viewport_width, const int viewport_height, glm::mat4 relativeTransformation, const int i = 0) {
            renderPointcloud(points, texture, model, view, projection, viewport_width, viewport_height, -1, -1, relativeTransformation, i);
        }

        void renderPointcloud(const rs2::points points, const rs2::frame texture, glm::mat4 model, glm::mat4 view, glm::mat4 projection, 
            const int viewport_width, const int viewport_height, const int pos_x, const int pos_y, glm::mat4 relativeTransformation = glm::mat4(1.0f), const int i = 0) {
            setViewport(viewport_width, viewport_height, pos_x, pos_y);

            renderCoordinateSystem(model, view, projection);
            //return;

            const rs2::vertex* vertices = points.get_vertices();
            const rs2::texture_coordinate* texCoords = points.get_texture_coordinates();
            const int num_vertices = points.size();

            //int num_not_zero_texCoords = 0;
            //std::stringstream ss;
            //for (int i = 0; i < num_vertices; i++)
            ////for (int i = num_vertices; i > 0; i--)
            //{
            //    rs2::texture_coordinate coord = texCoords[i];
            //    if (coord.u > 0 || coord.v > 0) {
            //        ss << coord.u << ", " << coord.v << "-";
            //        num_not_zero_texCoords++;
            //    }
            //    //ss << coord.u << ", " << coord.v << "-";
            //}
            //std::cout << ss.str() << std::endl;
            //std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;

            POINTCLOUD_shader->use();

            const int width = texture.as<rs2::video_frame>().get_width();
            const int height = texture.as<rs2::video_frame>().get_height();
            glBindTexture(GL_TEXTURE_2D, textures[0]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture.get_data());

            glBindVertexArray(VAOs[1]);
            
            glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
            glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(rs2::vertex), vertices, GL_STREAM_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(rs2::vertex), (void*)0);
            glEnableVertexAttribArray(0);

            glBindBuffer(GL_ARRAY_BUFFER, VBOs[2]);
            glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(rs2::texture_coordinate), texCoords, GL_STREAM_DRAW);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(rs2::texture_coordinate), (void*)0);
            glEnableVertexAttribArray(1);

            //auto color = DEBUG_COLORS[i];
            //POINTCLOUD_shader->setColor("color", color.r, color.g, color.b, 1);
            POINTCLOUD_shader->setColor("color", 1, 1, 1, 1);
            POINTCLOUD_shader->setMat4("relativeTransformation", relativeTransformation);
            POINTCLOUD_shader->setMat4("correction", COORDINATE_CORRECTION);
            POINTCLOUD_shader->setMat4("model", model);
            POINTCLOUD_shader->setMat4("view", view);
            POINTCLOUD_shader->setMat4("projection", projection);

            glDrawArrays(GL_POINTS, 0, num_vertices);
            glBindVertexArray(0);
        }

        void renderTexture(rs2::frame color_image, const float pos_x, const float pos_y, const float aspect, const int viewport_width, const int viewport_height) {
            setViewport(viewport_width, viewport_height, pos_x, pos_y);

            const int width = color_image.as<rs2::video_frame>().get_width();
            const int height = color_image.as<rs2::video_frame>().get_height();
            glBindTexture(GL_TEXTURE_2D, textures[0]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, color_image.get_data());

            glBindTexture(GL_TEXTURE_2D, textures[0]);

            const float image_aspect = 1.0f * width / height;

            float x_aspect = 1;
            float y_aspect = 1;
            if (image_aspect < aspect) {
                x_aspect = image_aspect / aspect;
            }
            else {
                y_aspect = aspect / image_aspect;
            }

            TEXTURE_shader->use();
            TEXTURE_shader->setVec2("aspect", x_aspect, y_aspect );
            glBindVertexArray(VAOs[0]);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);

           /* glBindBuffer(GL_ARRAY_BUFFER, COLOR_VBO);
            glBufferData(GL_ARRAY_BUFFER, vertices.size(), vertices.data(), GL_STREAM_DRAW);*/
        }

        void renderCoordinateSystem(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
            glEnable(GL_DEPTH_TEST);
            COORDINATE_SYSTEM_shader->use();
            COORDINATE_SYSTEM_shader->setMat4("model", model);
            COORDINATE_SYSTEM_shader->setMat4("view", view);
            COORDINATE_SYSTEM_shader->setMat4("projection", projection);
            glBindVertexArray(COORDINATE_SYSTEM_VAO);
            glDrawArrays(GL_TRIANGLES, 0, 9);
            glBindVertexArray(0);
            glDisable(GL_DEPTH_TEST);
        }

    };
    
    void startFrame(GLFWwindow* window) {
        glfwMakeContextCurrent(window);
        // render
        // ------
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        //glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
    }

    void setViewport(const int viewport_width, const int viewport_height, const int pos_x, const int pos_y) {
        if (pos_x < 0 || pos_y < 0) {
            glViewport(0, 0, viewport_width, viewport_height);
            return;
        }
        int width = viewport_width / 2;
        int height = viewport_height / 2;

        glViewport(width * pos_x, height - height * pos_y, width, height);
    }
}

#endif // !_RENDERING_HEADER_
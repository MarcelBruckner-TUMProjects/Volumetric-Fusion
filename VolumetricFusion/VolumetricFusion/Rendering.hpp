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

// disable compiler warning C4996
#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>

namespace vc::rendering {

#pragma region opengl_rendering
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

    glm::vec2* vertices;
    int num_vertices = 0;

    class Rendering {
    private:
        unsigned int COORDINATE_SYSTEM_VBO, COORDINATE_SYSTEM_VAO;
        unsigned int VBOs[3], VAOs[2], EBOs[1], textures[3];
        vc::rendering::Shader* TEXTURE_shader;
        vc::rendering::Shader* POINTCLOUD_shader;
        vc::rendering::Shader* POINTCLOUD_new_shader;
        vc::rendering::Shader* COORDINATE_SYSTEM_shader;

        std::vector<int> types = { GL_UNSIGNED_BYTE, GL_BYTE, GL_UNSIGNED_SHORT, GL_SHORT, GL_UNSIGNED_INT, GL_INT, GL_HALF_FLOAT, GL_FLOAT, GL_UNSIGNED_BYTE_3_3_2,
            GL_UNSIGNED_BYTE_2_3_3_REV, GL_UNSIGNED_SHORT_5_6_5, GL_UNSIGNED_SHORT_5_6_5_REV, GL_UNSIGNED_SHORT_4_4_4_4, GL_UNSIGNED_SHORT_4_4_4_4_REV,
            GL_UNSIGNED_SHORT_5_5_5_1, GL_UNSIGNED_SHORT_1_5_5_5_REV, GL_UNSIGNED_INT_8_8_8_8, GL_UNSIGNED_INT_8_8_8_8_REV, GL_UNSIGNED_INT_10_10_10_2, GL_UNSIGNED_INT_2_10_10_10_REV };

    public:
        int currentType = 0;
        Rendering() {
            TEXTURE_shader = new vc::rendering::VertexFragmentShader("shader/texture.vs", "shader/texture.fs");
            POINTCLOUD_shader = new vc::rendering::VertexFragmentShader("shader/pointcloud.vs", "shader/pointcloud.frag");
            POINTCLOUD_new_shader = new vc::rendering::VertexFragmentShader("shader/pointcloud_new.vert", "shader/pointcloud.frag");
            COORDINATE_SYSTEM_shader = new vc::rendering::VertexFragmentShader("shader/coordinate.vs", "shader/coordinate.fs");

            glGenVertexArrays(2, VAOs);
            glGenVertexArrays(1, &COORDINATE_SYSTEM_VAO);
            glGenBuffers(3, VBOs);
            glGenBuffers(1, &COORDINATE_SYSTEM_VBO);
            glGenBuffers(1, EBOs);
            glGenTextures(3, textures);

            for (int i = 0; i < 3; i++) {
                glBindTexture(GL_TEXTURE_2D, textures[i]); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
                // set the texture wrapping parameters
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                // set texture filtering parameters
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            }

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
        }

        void initializeVerticesBuffer() {
            glBindVertexArray(VAOs[1]);
            glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
            glBindBuffer(GL_ARRAY_BUFFER, VBOs[2]);
        }

        void renderAllPointclouds(const rs2::frame depth_frame, const rs2::frame color_frame, std::shared_ptr<vc::data::Camera> depth_camera, std::shared_ptr < vc::data::Camera> rgb_camera, 
            glm::mat4 model, glm::mat4 view, glm::mat4 projection, const int viewport_width, const int viewport_height, glm::mat4 relativeTransformation, const int i = 0, bool renderCoordinateSystem = false) {
            renderPointcloud(depth_frame, color_frame, depth_camera, rgb_camera, model, view, projection, viewport_width, viewport_height, -1, -1, relativeTransformation, i, renderCoordinateSystem);
        }

        void renderPointcloud(const rs2::frame depth_frame, const rs2::frame color_frame, std::shared_ptr<vc::data::Camera> depth_camera, std::shared_ptr < vc::data::Camera> rgb_camera,
            glm::mat4 model, glm::mat4 view, glm::mat4 projection, const int viewport_width, const int viewport_height, const int pos_x, const int pos_y, glm::mat4 relativeTransformation = glm::mat4(1.0f), const int i = 0, bool renderCoordinateSystem = false) {
            setViewport(viewport_width, viewport_height, pos_x, pos_y);

            if (renderCoordinateSystem) {
                this->renderCoordinateSystem(model, view, projection);
            }

            //glEnable(GL_DEPTH_TEST);

            POINTCLOUD_new_shader->use();

            int width = color_frame.as<rs2::video_frame>().get_width();
            int height = color_frame.as<rs2::video_frame>().get_height();
            int color_size = color_frame.get_data_size();
            
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, textures[0]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, color_frame.get_data());
            POINTCLOUD_new_shader->setInt("color_frame", 1);

            width = depth_frame.as<rs2::video_frame>().get_width();
            height = depth_frame.as<rs2::video_frame>().get_height();

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textures[1]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, depth_frame.get_data());
            POINTCLOUD_new_shader->setInt("depth_frame", 0);

            int current_num_vertices = width * height;
            if (current_num_vertices != num_vertices) {
                num_vertices = current_num_vertices;
                delete[] vertices;
                vertices = new glm::vec2[num_vertices];
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        vertices[y * width + x] = glm::vec2(1.0f * x / width, 1.0f * y / height);
                    }
                }
            }

            glBindVertexArray(VAOs[1]);

            glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
            glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(glm::vec2), vertices, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);
            glEnableVertexAttribArray(0);

            //POINTCLOUD_new_shader->set

            POINTCLOUD_new_shader->setMat3("cam2world", depth_camera->cam2world);
            POINTCLOUD_new_shader->setFloat("depth_scale", depth_camera->depthScale);
            POINTCLOUD_new_shader->setFloat("aspect", 1.0f * width / height);

            POINTCLOUD_new_shader->setMat4("relativeTransformation", relativeTransformation);
            POINTCLOUD_new_shader->setMat4("correction", COORDINATE_CORRECTION);
            POINTCLOUD_new_shader->setMat4("model", model);
            POINTCLOUD_new_shader->setMat4("view", view);
            POINTCLOUD_new_shader->setMat4("projection", projection);

            glDrawArrays(GL_POINTS, 0, num_vertices);
            glBindVertexArray(0);
        }
    
        //void renderPointcloud(const rs2::points points, const rs2::frame texture, glm::mat4 model, glm::mat4 view, glm::mat4 projection,
        //        const int viewport_width, const int viewport_height, const int pos_x, const int pos_y, glm::mat4 relativeTransformation = glm::mat4(1.0f), const int i = 0) {
        //    setViewport(viewport_width, viewport_height, pos_x, pos_y);

        //    renderCoordinateSystem(model, view, projection);
        //    //return;

        //    const rs2::vertex* vertices = points.get_vertices();
        //    const rs2::texture_coordinate* texCoords = points.get_texture_coordinates();
        //    const int num_vertices = points.size();

        //    //int num_not_zero_texCoords = 0;
        //    //std::stringstream ss;
        //    //for (int i = 0; i < num_vertices; i++)
        //    ////for (int i = num_vertices; i > 0; i--)
        //    //{
        //    //    rs2::texture_coordinate coord = texCoords[i];
        //    //    if (coord.u > 0 || coord.v > 0) {
        //    //        ss << coord.u << ", " << coord.v << "-";
        //    //        num_not_zero_texCoords++;
        //    //    }
        //    //    //ss << coord.u << ", " << coord.v << "-";
        //    //}
        //    //std::cout << ss.str() << std::endl;
        //    //std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
        //    glEnable(GL_DEPTH_TEST);

        //    POINTCLOUD_shader->use();

        //    const int width = texture.as<rs2::video_frame>().get_width();
        //    const int height = texture.as<rs2::video_frame>().get_height();
        //    glBindTexture(GL_TEXTURE_2D, textures[0]);
        //    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture.get_data());

        //    glBindVertexArray(VAOs[1]);
        //    
        //    glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
        //    glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(rs2::vertex), vertices, GL_STREAM_DRAW);
        //    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(rs2::vertex), (void*)0);
        //    glEnableVertexAttribArray(0);

        //    glBindBuffer(GL_ARRAY_BUFFER, VBOs[2]);
        //    glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(rs2::texture_coordinate), texCoords, GL_STREAM_DRAW);
        //    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(rs2::texture_coordinate), (void*)0);
        //    glEnableVertexAttribArray(1);

        //    //auto color = DEBUG_COLORS[i];
        //    //POINTCLOUD_shader->setColor("color", color.r, color.g, color.b, 1);
        //    POINTCLOUD_shader->setColor("color", 1, 1, 1, 1);
        //    POINTCLOUD_shader->setMat4("relativeTransformation", relativeTransformation);
        //    POINTCLOUD_shader->setMat4("correction", COORDINATE_CORRECTION);
        //    POINTCLOUD_shader->setMat4("model", model);
        //    POINTCLOUD_shader->setMat4("view", view);
        //    POINTCLOUD_shader->setMat4("projection", projection);

        //    glDrawArrays(GL_POINTS, 0, num_vertices);
        //    glBindVertexArray(0);
        //}

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
            COORDINATE_SYSTEM_shader->setMat4("correction", COORDINATE_CORRECTION);
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

#pragma endregion

    class SurfaceRendering {
    private:
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr;
        pcl::PointCloud<pcl::Normal>::Ptr normals_ptr;

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals_ptr;

        pcl::PolygonMesh::Ptr mesh_ptr;

    public:
        SurfaceRendering(const float* points, const int pointCount) {
            std::uint32_t rgb_color_green(0x00FF00);
            std::uint32_t rgb_color_red(0xFF0000);
            std::uint32_t rgb_color_blue(0x0000FF);
            std::uint32_t rgb_color_yellow(0xFFFF00);

            pcl::console::TicToc tt;

            // https://github.com/IntelRealSense/librealsense/blob/master/wrappers/pcl/pcl/rs-pcl.cpp
            std::cout << "Constructing point cloud ... ";
            tt.tic();
            this->cloud_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
            this->cloud_ptr->width = static_cast<std::uint32_t>(pointCount / 3);
            this->cloud_ptr->height = 1;
            this->cloud_ptr->is_dense = true;
            this->cloud_ptr->points.reserve(this->cloud_ptr->width * this->cloud_ptr->height);
            for (int i = 0; i < pointCount; i += 3) {
                pcl::PointXYZRGB point;
                point.x = points[i + 0];
                point.y = points[i + 1];
                point.z = points[i + 2];
                point.rgb = *reinterpret_cast<float*>(&rgb_color_yellow);
                this->cloud_ptr->points.push_back(point);
            }
            std::cout << "done in " << tt.toc() << "ms" << std::endl;

            // Downsampling: http://pointclouds.org/documentation/tutorials/voxel_grid.php
            pcl::VoxelGrid<pcl::PCLPointCloud2> vg;
            std::cout << "Downsampling ... ";
            auto oldCloudSize = this->cloud_ptr->points.size();
            tt.tic();
            pcl::PCLPointCloud2::Ptr cloud_ptr2(new pcl::PCLPointCloud2);
            pcl::toPCLPointCloud2(*this->cloud_ptr, *cloud_ptr2);
            pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());
            vg.setInputCloud(cloud_ptr2);
            vg.setLeafSize(0.01f, 0.01f, 0.01f);
            vg.filter(*cloud_filtered);
            pcl::fromPCLPointCloud2(*cloud_filtered, *this->cloud_ptr);
            std::cout << "done in " << tt.toc() << "ms (was=" << oldCloudSize << ", is=" << this->cloud_ptr->points.size() << ")" << std::endl;

            // http://pointclouds.org/documentation/tutorials/resampling.php
            std::cout << "Smoothing ... ";
            tt.tic();
            pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
            mls.setInputCloud(this->cloud_ptr);
            mls.setSearchRadius(0.01);
            mls.setPolynomialFit(true);
            mls.setPolynomialOrder(2);
            mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB>::SAMPLE_LOCAL_PLANE);
            mls.setUpsamplingRadius(0.005);
            mls.setUpsamplingStepSize(0.003);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZRGB>());
            mls.process(*cloud_smoothed);
            this->cloud_ptr = cloud_smoothed;
            std::cout << "done in " << tt.toc() << "ms" << std::endl;

            // Transformations?
            //Eigen::Matrix4f transform0 = Eigen::Matrix4f::Identity();
            //glm::mat4 rt0 = relativeTransformations[1];
            //transform0(0, 0) = rt0[0][0]; transform0(0, 1) = rt0[0][1]; transform0(0, 2) = rt0[0][2]; transform0(0, 3) = rt0[0][3];
            //transform0(1, 0) = rt0[1][0]; transform0(1, 1) = rt0[1][1]; transform0(1, 2) = rt0[1][2]; transform0(1, 3) = rt0[1][3];
            //transform0(2, 0) = rt0[2][0]; transform0(2, 1) = rt0[2][1]; transform0(2, 2) = rt0[2][2]; transform0(2, 3) = rt0[2][3];
            //transform0(3, 0) = rt0[3][0]; transform0(3, 1) = rt0[3][1]; transform0(3, 2) = rt0[3][2]; transform0(3, 3) = rt0[3][3];
            //std::cout << "Transform1: " << transform0 << ", glm1: " << glm::to_string(rt0) << std::endl;

            std::cout << "Computing normals ... ";
            tt.tic();
            // Create the normal estimation class, and pass the input dataset to it
            //pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
            pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setInputCloud(this->cloud_ptr);
            // Create an empty kdtree representation, and pass it to the normal estimation object.
            // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
            ne.setSearchMethod(tree);
            // Output datasets
            this->normals_ptr = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
            // Use all neighbors in a sphere of radius 3cm
            ne.setRadiusSearch(0.01);
            // Compute the features
            ne.compute(*this->normals_ptr);
            // maybe use computePointNormal instead since the point cloud should be ordered and therefore the nearest neighbors 
            // should be known
            std::cout << "done in " << tt.toc() << "ms (size: " << normals_ptr->points.size() << " =? " << cloud_ptr->points.size() << ")" << std::endl;

            std::cout << "Combining point cloud and normals ... ";
            tt.tic();
            // Initialization part
            this->cloud_normals_ptr = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            this->cloud_normals_ptr->width = this->cloud_ptr->width;
            this->cloud_normals_ptr->height = this->cloud_ptr->height;
            this->cloud_normals_ptr->is_dense = true;
            this->cloud_normals_ptr->points.reserve(this->cloud_normals_ptr->width * this->cloud_normals_ptr->height);
            // Assignment part
            for (int i = 0; i < this->normals_ptr->points.size(); i++)
            { 
                pcl::PointXYZRGBNormal cn_point;
                cn_point.x = this->cloud_ptr->points[i].x;
                cn_point.y = this->cloud_ptr->points[i].y;
                cn_point.z = this->cloud_ptr->points[i].z;
                cn_point.r = this->cloud_ptr->points[i].r;
                cn_point.g = this->cloud_ptr->points[i].g;
                cn_point.b = this->cloud_ptr->points[i].b;
                cn_point.curvature = this->normals_ptr->points[i].curvature;
                cn_point.normal_x = this->normals_ptr->points[i].normal_x;
                cn_point.normal_y = this->normals_ptr->points[i].normal_y;
                cn_point.normal_z = this->normals_ptr->points[i].normal_z;
                this->cloud_normals_ptr->points.push_back(cn_point);
            }
            std::cout << "done in " << tt.toc() << "ms" << std::endl;

            std::cout << "Running Marching Cubes (Hoppe) ... ";
            pcl::MarchingCubes<pcl::PointXYZRGBNormal>* mc = new pcl::MarchingCubesHoppe<pcl::PointXYZRGBNormal>();
            mc->setIsoLevel(0.001f);
            //mc->setGridResolution(100, 100, 100);
            mc->setGridResolution(25, 25, 25);
            mc->setPercentageExtendGrid(0.0f);
            mc->setInputCloud(this->cloud_normals_ptr);
            tt.tic();
            this->mesh_ptr = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh);
            mc->reconstruct(*this->mesh_ptr);
            std::cout << "done in " << tt.toc() << "ms" << std::endl;

            //tt.tic();
            //std::cout << "Savinng to 'rendered_cloud.pcd'";
            //pcl::io::savePCDFile("rendered_cloud.pcd", *this->cloud_ptr);
            //std::cout << ", 'rendered_cloud_normals.pcd'";
            //pcl::io::savePCDFile("rendered_cloud_normals.pcd", *this->cloud_normals_ptr);
            //std::cout << ", 'rendered.vtk' ";
            //pcl::io::saveVTKFile("rendered_mesh.vtk", *this->mesh_ptr);
            //std::cout << " and 'rendered_cloud_normals.ply ...' ";
            //pcl::io::savePLYFileBinary("rendered_cloud_normals.ply", *this->cloud_normals_ptr);
            //std::cout << "done in " << tt.toc() << "ms" << std::endl;
        }

        ~SurfaceRendering() {
            // delete anything?
        }

        void render() {
            std::cout << "Rendering ... ";
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
            //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_blue(this->cloud_ptr);
            //viewer->addPointCloud<pcl::PointXYZRGB>(this->cloud_ptr, rgb_blue, "pc 0");
            //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "pc 0");
            viewer->setBackgroundColor(0.8f, 0.8f, 0.8f);
            viewer->addCoordinateSystem(0.1);
            viewer->initCameraParameters();
            //viewer->registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);

            // bug fix for: viewer->addPolygonMesh(*this->mesh_ptr, "polygon_o");
            // https://github.com/PointCloudLibrary/pcl/issues/178#issuecomment-327410610
            vtkSmartPointer<vtkPolyData> poly_data;
            pcl::VTKUtils::mesh2vtk(*this->mesh_ptr, poly_data);
            viewer->addModelFromPolyData(poly_data, "poly_data", 0);
            viewer->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_SHADING,
                pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, 
                "poly_data"
            );
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.5, 0.5, 0, "poly_data");
            viewer->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
                pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, 
                "poly_data"
            );

            //while (!viewer->wasStopped()) {
            //    viewer->spinOnce(100);
            //    std::this_thread::sleep_for(100ms);
            //}
            std::cout << "done" << std::endl;
        }
    };
}

#endif
#pragma once

#ifndef _RENDERING_HEADER_
#define _RENDERING_HEADER_

#if KEVIN_MACOS
#pragma message("Included on Mac OS")
#endif

#include <librealsense2/rs.hpp>

//#include <GLFW/glfw3.h>

#define GL_GLEXT_PROTOTYPES

//#include <GL/GL.h>

#include <librealsense2/rs.hpp> 

#include <string>
#include <sstream>
#include <iostream>

namespace vc::rendering {

	class Rendering {
	private:
		GLFWwindow* window;
		glfw_state* app_state;

		void gl_render_cube(float width, float height, float x, float y, float z) {

		}

	public:
		Rendering(GLFWwindow *window, glfw_state* app_sate)
			: window(window), app_state(app_state)
		{
		}

		void test() {
			glPushMatrix();

			// Initial scene settings
			glLoadIdentity();
			glPushAttrib(GL_ALL_ATTRIB_BITS);
			// Reset the background to be black
			glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
			glClear(GL_DEPTH_BUFFER_BIT);
			//glDisable(GL_DEPTH_TEST);

			glMatrixMode(GL_PROJECTION);
			//gluPerspective(60, width / height, 0.01f, 10.0f);

			glPopMatrix();
			glEnd();
		}
	};

    void draw_all_vertices_and_colors(
            float width, float height,
            glfw_state& app_state,
            std::vector<std::shared_ptr<  vc::capture::CaptureDevice>> pipelines,
            std::map<int, Eigen::MatrixXd> relativeTransformations,
            GLFWwindow* window = nullptr
    ) {
        if (window != nullptr) {
            glfwMakeContextCurrent(window);
        }

        // OpenGL commands that prep screen for the pointcloud
        glLoadIdentity();
        glPushAttrib(GL_ALL_ATTRIB_BITS);

        glClearColor(0.5f, 0.5f, 0.5f , 1); 
		glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        gluPerspective(60, width / height, 0.01f, 10.0f);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

        for (int i = 0; i < pipelines.size(); ++i) {
            auto transformation = relativeTransformations[i];
            auto color_frame = pipelines[i]->data->filteredColorFrames;
            auto points = pipelines[i]->data->points;

            const double* tdata = transformation.data();
            glLoadMatrixd(tdata);

			gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

			glTranslatef(0, 0, +0.5f + app_state.offset_y * 0.05f);
			glRotated(app_state.pitch, 1, 0, 0);
			glRotated(app_state.yaw, 0, 1, 0);
			glTranslatef(0, 0, -0.5f);

            glPointSize(width / 640);
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            //glEnable(GL_DEPTH_TEST);
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, app_state.tex.get_gl_handle());

            // print cube at center


            // print the colors
            auto format = color_frame.get_profile().format();
            switch (format)
            {
                case RS2_FORMAT_RGB8:
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, color_frame.get_data());
                    break;
                case RS2_FORMAT_RGBA8:
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, color_frame.get_data());
                    break;
                case RS2_FORMAT_Y8:
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, color_frame.get_data());
                    break;
                case RS2_FORMAT_Y10BPACK:
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT, color_frame.get_data());
                    break;
                default:
                    throw std::runtime_error("The requested format is not supported!");
            }

            //float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
            //glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE
            glBegin(GL_POINTS);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glBindTexture(GL_TEXTURE_2D, 0);

            /* this segment actually prints the pointcloud */
            auto vertices = points.get_vertices();              // get vertices
            auto tex_coords = points.get_texture_coordinates(); // and texture coordinates
            for (int i = 0; i < points.size(); i++)
            {
                if (vertices[i].z)
                {
                    // upload the point and texture coordinates only for points we have depth data for
                    glVertex3fv(vertices[i]);
                    glTexCoord2fv(tex_coords[i]);
                }
            }
            glPopMatrix();
        }

        // OpenGL cleanup
        glEnd();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glPopAttrib();
    }

	void draw_vertices_and_colors(
		float width, float height, glfw_state& app_state, 
		rs2::points& points, 
		rs2::frame color_frame,
		const Eigen::Ref<const Eigen::MatrixXd>& transformation,
		GLFWwindow* window = nullptr
	) {
		if (window != nullptr) {
			glfwMakeContextCurrent(window);
		}
		if (!points)
			return;

		// OpenGL commands that prep screen for the pointcloud
		glLoadIdentity();
		glPushAttrib(GL_ALL_ATTRIB_BITS);

		glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
		glClear(GL_DEPTH_BUFFER_BIT);
		glDisable(GL_DEPTH_TEST);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		gluPerspective(60, width / height, 0.01f, 10.0f);

		if (transformation.rows() > 0) {
			const double* tdata = transformation.data();
			glLoadMatrixd(tdata);
		}

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

		glTranslatef(0, 0, +0.5f + app_state.offset_y * 0.05f);
		glRotated(app_state.pitch, 1, 0, 0);
		glRotated(app_state.yaw, 0, 1, 0);
		glTranslatef(0, 0, -0.5f);

		glPointSize(width / 640);
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		//glEnable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, app_state.tex.get_gl_handle());

		// print cube at center


		// print the colors
		auto format = color_frame.get_profile().format();
		switch (format)
		{
		case RS2_FORMAT_RGB8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, color_frame.get_data());
			break;
		case RS2_FORMAT_RGBA8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, color_frame.get_data());
			break;
		case RS2_FORMAT_Y8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, color_frame.get_data());
			break;
		case RS2_FORMAT_Y10BPACK:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT, color_frame.get_data());
			break;
		default:
			throw std::runtime_error("The requested format is not supported!");
		}

		//float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
		//glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE
		glBegin(GL_POINTS);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		/* this segment actually prints the pointcloud */
		auto vertices = points.get_vertices();              // get vertices
		auto tex_coords = points.get_texture_coordinates(); // and texture coordinates
		for (int i = 0; i < points.size(); i++)
		{
			if (vertices[i].z)
			{
				// upload the point and texture coordinates only for points we have depth data for
				glVertex3fv(vertices[i]);
				glTexCoord2fv(tex_coords[i]);
			}
		}

		// OpenGL cleanup
		glEnd();
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glPopAttrib();
	}


	void draw_rectangle(float width, float height, float x, float y, float z, glfw_state& app_state, const Eigen::Ref<const Eigen::MatrixXd>& transformation, GLFWwindow* window = nullptr) {
		if (window != nullptr) {
			glfwMakeContextCurrent(window);
		}

		// OpenGL commands that prep screen for the pointcloud
		glLoadIdentity();
		glPushAttrib(GL_ALL_ATTRIB_BITS);

		//glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
		//glClear(GL_DEPTH_BUFFER_BIT);






		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		gluPerspective(60, width / height, 0.01f, 10.0f);

		if (transformation.rows() > 0) {
			const double* tdata = transformation.data();
			glLoadMatrixd(tdata);
		}

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

		glTranslatef(x, y, z + 0.5f + app_state.offset_y * 0.05f);
		//glTranslatef(x,y,z);
		glRotated(app_state.pitch, 1, 0, 0);
		glRotated(app_state.yaw, 0, 1, 0);







		glScalef(0.02, 0.02, 0.02);

		glPointSize(width / 640);
		glEnable(GL_DEPTH_TEST);

		glBegin(GL_QUADS);        // Draw The Cube Using quads
		glColor3f(0.0f, 1.0f, 0.0f);    // Color Blue
		glVertex3f(1.0f, 1.0f, -1.0f);    // Top Right Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f, -1.0f);    // Top Left Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f, 1.0f);    // Bottom Left Of The Quad (Top)
		glVertex3f(1.0f, 1.0f, 1.0f);    // Bottom Right Of The Quad (Top)
		glColor3f(1.0f, 0.5f, 0.0f);    // Color Orange
		glVertex3f(1.0f, -1.0f, 1.0f);    // Top Right Of The Quad (Bottom)
		glVertex3f(-1.0f, -1.0f, 1.0f);    // Top Left Of The Quad (Bottom)
		glVertex3f(-1.0f, -1.0f, -1.0f);    // Bottom Left Of The Quad (Bottom)
		glVertex3f(1.0f, -1.0f, -1.0f);    // Bottom Right Of The Quad (Bottom)
		glColor3f(1.0f, 0.0f, 0.0f);    // Color Red
		glVertex3f(1.0f, 1.0f, 1.0f);    // Top Right Of The Quad (Front)
		glVertex3f(-1.0f, 1.0f, 1.0f);    // Top Left Of The Quad (Front)
		glVertex3f(-1.0f, -1.0f, 1.0f);    // Bottom Left Of The Quad (Front)
		glVertex3f(1.0f, -1.0f, 1.0f);    // Bottom Right Of The Quad (Front)
		glColor3f(1.0f, 1.0f, 0.0f);    // Color Yellow
		glVertex3f(1.0f, -1.0f, -1.0f);    // Top Right Of The Quad (Back)
		glVertex3f(-1.0f, -1.0f, -1.0f);    // Top Left Of The Quad (Back)
		glVertex3f(-1.0f, 1.0f, -1.0f);    // Bottom Left Of The Quad (Back)
		glVertex3f(1.0f, 1.0f, -1.0f);    // Bottom Right Of The Quad (Back)
		glColor3f(0.0f, 0.0f, 1.0f);    // Color Blue
		glVertex3f(-1.0f, 1.0f, 1.0f);    // Top Right Of The Quad (Left)
		glVertex3f(-1.0f, 1.0f, -1.0f);    // Top Left Of The Quad (Left)
		glVertex3f(-1.0f, -1.0f, -1.0f);    // Bottom Left Of The Quad (Left)
		glVertex3f(-1.0f, -1.0f, 1.0f);    // Bottom Right Of The Quad (Left)
		glColor3f(1.0f, 0.0f, 1.0f);    // Color Violet
		glVertex3f(1.0f, 1.0f, -1.0f);    // Top Right Of The Quad (Right)
		glVertex3f(1.0f, 1.0f, 1.0f);    // Top Left Of The Quad (Right)
		glVertex3f(1.0f, -1.0f, 1.0f);    // Bottom Left Of The Quad (Right)
		glVertex3f(1.0f, -1.0f, -1.0f);    // Bottom Right Of The Quad (Right)
		glEnd();            // End Drawing The Cube
		glFlush();



		// OpenGL cleanup
		glEnd();
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glPopAttrib();
	}

	// Working copy
	void _draw_vertices_and_colors(
		float width, float height, glfw_state& app_state,
		rs2::points& points,
		rs2::frame color_frame,
		const Eigen::Ref<const Eigen::MatrixXd>& transformation,
		GLFWwindow* window = nullptr
	) {
		if (window != nullptr) {
			glfwMakeContextCurrent(window);
		}
		if (!points)
			return;

		// OpenGL commands that prep screen for the pointcloud
		glLoadIdentity();
		glPushAttrib(GL_ALL_ATTRIB_BITS);

		glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
		glClear(GL_DEPTH_BUFFER_BIT);
		glDisable(GL_DEPTH_TEST);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		gluPerspective(60, width / height, 0.01f, 10.0f);

		if (transformation.rows() > 0) {
			const double* tdata = transformation.data();
			glLoadMatrixd(tdata);
		}

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

		glTranslatef(0, 0, +0.5f + app_state.offset_y * 0.05f);
		glRotated(app_state.pitch, 1, 0, 0);
		glRotated(app_state.yaw, 0, 1, 0);
		glTranslatef(0, 0, -0.5f);

		glPointSize(width / 640);
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		//glEnable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, app_state.tex.get_gl_handle());

		// print the colors
		auto format = color_frame.get_profile().format();
		switch (format)
		{
		case RS2_FORMAT_RGB8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, color_frame.get_data());
			break;
		case RS2_FORMAT_RGBA8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, color_frame.get_data());
			break;
		case RS2_FORMAT_Y8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, color_frame.get_data());
			break;
		case RS2_FORMAT_Y10BPACK:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT, color_frame.get_data());
			break;
		default:
			throw std::runtime_error("The requested format is not supported!");
		}

		//float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
		//glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE
		glBegin(GL_POINTS);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		/* this segment actually prints the pointcloud */
		auto vertices = points.get_vertices();              // get vertices
		auto tex_coords = points.get_texture_coordinates(); // and texture coordinates
		for (int i = 0; i < points.size(); i++)
		{
			if (vertices[i].z)
			{
				// upload the point and texture coordinates only for points we have depth data for
				glVertex3fv(vertices[i]);
				glTexCoord2fv(tex_coords[i]);
			}
		}

		// OpenGL cleanup
		glEnd();
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glPopAttrib();
	}
}

#endif // !_RENDERING_HEADER_

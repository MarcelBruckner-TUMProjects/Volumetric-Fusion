#pragma once

#ifndef _RENDERING_HEADER_
#define _RENDERING_HEADER_

#if KEVIN_MACOS
#pragma message("Included on Mac OS")
#endif

#define GL_GLEXT_PROTOTYPES

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <set>
#include <librealsense2/rs.hpp>
//#define GL_SILENCE_DEPRECATION
#include <glad/glad.h>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

namespace vc::rendering {

	struct MouseState {
		cv::Vec2f pos;

		bool leftPressed{ false };
		bool rightPressed{ false };
	};


	class ViewerInput {
	public:
		static ViewerInput& get() // Singleton is accessed via get()
		{
			static ViewerInput instance; // lazy singleton, instantiated on first use
			return instance;
		}

		// Callback methods.
		static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
			get().mouseButtonCallbackImpl(button, action);
		}

		void mouseButtonCallbackImpl(int button, int action) {
			if (button == GLFW_MOUSE_BUTTON_1) {
				if (action == GLFW_PRESS) {
					m_bMouseLeftDown = true;
					m_mouse.leftPressed = true;
				}
				else if (action == GLFW_RELEASE) {
					m_bMouseLeftUp = true;
					m_mouse.leftPressed = false;
				}
			}
			if (button == GLFW_MOUSE_BUTTON_1) {
				if (action == GLFW_PRESS) {
					m_bMouseLeftDown = true;
					m_mouse.rightPressed = true;
				}
				else if (action == GLFW_RELEASE) {
					m_bMouseLeftUp = true;
					m_mouse.rightPressed = false;
				}
			}
		}

		static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
			if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
				glfwSetWindowShouldClose(window, 1);

			//here we access the instance via the singleton pattern and forward the callback to the instance method
			get().keyCallbackImpl(key, action);
		}

		void keyCallbackImpl(int key, int action) {
			if (action == GLFW_PRESS) {
				m_keysPressed[key] = true;
				m_keysDown.insert(key);
			}
			else if (action == GLFW_RELEASE) {
				m_keysPressed[key] = false;
				m_keysUp.insert(key);
			}
		}

		static void scrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
			get().scrollCallbackImpl(xOffset, yOffset);
		}

		void scrollCallbackImpl(double xOffset, double yOffset) {
			m_scrollChangeX = xOffset;
			m_scrollChangeY = yOffset;
			m_bScrollChange = true;
		}

		static void cursorPosCallback(GLFWwindow* window, double x, double y) {
			get().cursorPosCallbackImpl(x, y);
		}

		void cursorPosCallbackImpl(double x, double y) {
			m_bMouseMoved = true;
			m_mouse.pos = cv::Vec2f(x, y);
		}

		static void errorCallback(int error, const char* description) {
			fprintf(stderr, "Error: %s\n", description);
		}

		// Resets change events.
		void resetChangeEvents() {
			m_keysUp.clear();
			m_keysDown.clear();

			m_bMouseMoved = false;
			m_bMouseLeftUp = false;
			m_bMouseLeftDown = false;
			m_bMouseRightUp = false;
			m_bMouseRightDown = false;
			m_bScrollChange = false;

			m_mousePrev = m_mouse;
			m_scrollChangeX = 0.0;
			m_scrollChangeY = 0.0;
		}

		// Event querries.
		bool isKeyDown(int key) const {
			return m_keysDown.find(key) != m_keysDown.end();
		}

		bool isKeyUp(int key) const {
			return m_keysUp.find(key) != m_keysUp.end();
		}

		bool isKeyPressed(int key) const {
			return m_keysPressed.find(key) != m_keysPressed.end() && m_keysPressed.find(key)->second;
		}

		bool isMouseLeftDown() const {
			return m_bMouseLeftDown;
		}

		bool isMouseLeftUp() const {
			return m_bMouseLeftUp;
		}

		bool isMouseMoved() const {
			return m_bMouseMoved;
		}

		bool isScrollChanged() const {
			return m_bScrollChange;
		}

		const MouseState& getMouse() const {
			return m_mouse;
		}

		const MouseState& getMousePrev() const {
			return m_mousePrev;
		}

		double getScrollChangeX() const {
			return m_scrollChangeX;
		}

		double getScrollChangeY() const {
			return m_scrollChangeY;
		}

	private:
		std::set<int> m_keysDown;
		std::set<int> m_keysUp;
		std::map<int, bool> m_keysPressed;

		MouseState m_mousePrev;
		MouseState m_mouse;

		double m_scrollChangeX{ 0.0 };
		double m_scrollChangeY{ 0.0 };

		bool m_bMouseMoved{ false };
		bool m_bMouseLeftUp{ false };
		bool m_bMouseLeftDown{ false };
		bool m_bMouseRightUp{ false };
		bool m_bMouseRightDown{ false };
		bool m_bScrollChange{ false };

		ViewerInput(void) // private constructor necessary to allow only 1 instance
		{ }

		ViewerInput(ViewerInput const&); // prevent copies
		void operator=(ViewerInput const&); // prevent assignments
	};

	class Rendering {
	private:
		GLFWwindow* window = nullptr;
		glfw_state* app_state;

		GLuint defaultProgram;
		unsigned int VBO, VAO, EBO;

		std::string readFile(const char* filePath) {
			std::string content;
			std::ifstream fileStream(filePath, std::ios::in);

			if (!fileStream.is_open()) {
				std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
				return "";
			}

			std::string line = "";
			while (!fileStream.eof()) {
				std::getline(fileStream, line);
				content.append(line + "\n");
			}

			fileStream.close();
			return content;
		}

		GLuint loadShader(const char* vertex_path, const char* fragment_path) {
			GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
			GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

			// Read shaders

			std::string vertShaderStr = readFile(vertex_path);
			std::string fragShaderStr = readFile(fragment_path);
			if (vertShaderStr.length() == 0 || fragShaderStr.length() == 0) {
				return 0;
			}
			const char* vertShaderSrc = vertShaderStr.c_str();
			const char* fragShaderSrc = fragShaderStr.c_str();

			GLint result = GL_FALSE;
			int logLength;

			// Compile vertex shader

			std::cout << "Compiling vertex shader" << std::endl;
			glShaderSource(vertShader, 1, &vertShaderSrc, NULL);
			glCompileShader(vertShader);

			// Check vertex shader

			glGetShaderiv(vertShader, GL_COMPILE_STATUS, &result);
			if (!result) {
				glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &logLength);
				std::vector<char> vertShaderError((logLength > 1) ? logLength : 1);
				glGetShaderInfoLog(vertShader, logLength, NULL, &vertShaderError[0]);
				std::cerr << &vertShaderError[0] << std::endl;
				return 0;
			}

			// Compile fragment shader

			std::cout << "Compiling fragment shader" << std::endl;
			glShaderSource(fragShader, 1, &fragShaderSrc, NULL);
			glCompileShader(fragShader);

			// Check fragment shader

			glGetShaderiv(fragShader, GL_COMPILE_STATUS, &result);
			if (!result) {
				glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &logLength);
				std::vector<char> fragShaderError((logLength > 1) ? logLength : 1);
				glGetShaderInfoLog(fragShader, logLength, NULL, &fragShaderError[0]);
				std::cerr << &fragShaderError[0] << std::endl;
				return 0;
			}

			std::cout << "Linking program" << std::endl;
			GLuint program = glCreateProgram();
			glAttachShader(program, vertShader);
			glAttachShader(program, fragShader);
			glLinkProgram(program);

			glGetProgramiv(program, GL_LINK_STATUS, &result);
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
			std::vector<char> programError((logLength > 1) ? logLength : 1);
			glGetProgramInfoLog(program, logLength, NULL, &programError[0]);
			std::cout << &programError[0] << std::endl;

			glDeleteShader(vertShader);
			glDeleteShader(fragShader);

			return program;
		}

		void printGLVersion() {
			const GLubyte* renderer = glGetString(GL_RENDERER);
			const GLubyte* vendor = glGetString(GL_VENDOR);
			const GLubyte* version = glGetString(GL_VERSION);
			const GLubyte* glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
			GLint major, minor;
			glGetIntegerv(GL_MAJOR_VERSION, &major);
			glGetIntegerv(GL_MINOR_VERSION, &minor);
			printf("GL Vendor : %s\n", vendor);
			printf("GL Renderer : %s\n", renderer);
			printf("GL Version (string) : %s\n", version);
			printf("GL Version (integer) : %d.%d\n", major, minor);
			printf("GLSL Version : %s\n", glslVersion);
		}

	public:
		Rendering(float screenWidth, float screenHeight, glfw_state* app_sate)
			: app_state(app_state)
		{
			/*GLenum err = glewInit();
			if (GLEW_OK != err) {
				fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
			}*/

			glfwSetErrorCallback([](int error, const char* description) {
				fprintf(stderr, "glfwError: %s\n", description);
			});
			if (!glfwInit()) {
				exit(EXIT_FAILURE);
			}
			glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
			glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

			window = glfwCreateWindow(screenWidth, screenHeight, "Visualization", NULL, NULL);
			if (!window) {
				glfwTerminate();
				exit(EXIT_FAILURE);
			}

			// Set event callbacks.
			glfwSetKeyCallback(window, ViewerInput::keyCallback);
			glfwSetMouseButtonCallback(window, ViewerInput::mouseButtonCallback);
			glfwSetScrollCallback(window, ViewerInput::scrollCallback);
			glfwSetCursorPosCallback(window, ViewerInput::cursorPosCallback);

			glfwMakeContextCurrent(window);
			gladLoadGL();
			glfwSwapInterval(1);

			this->printGLVersion();

			defaultProgram = loadShader("shaders/default.vert", "shaders/default.frag");

			float vertices[] = {
				 0.5f,  0.5f, 0.0f,  // top right
				 0.5f, -0.5f, 0.0f,  // bottom right
				-0.5f, -0.5f, 0.0f,  // bottom left
				-0.5f,  0.5f, 0.0f   // top left 
			};
			unsigned int indices[] = {  // note that we start from 0!
				0, 1, 3,   // first triangle
				1, 2, 3    // second triangle
			};

			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);
			glGenBuffers(1, &EBO);

			// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
			// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
			glBindVertexArray(0);

			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}

		void opengl4_draw_all_vertices_and_colors(
			float width, float height,
			glfw_state& app_state,
			std::vector<std::shared_ptr<  vc::capture::CaptureDevice>> pipelines,
			std::map<int, Eigen::MatrixXd> relativeTransformations,
			GLFWwindow* window = nullptr
		) {
			glfwMakeContextCurrent(this->window);

			glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);

			glUseProgram(defaultProgram);
			glBindVertexArray(VAO);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);

			glfwSwapBuffers(this->window);
			glfwPollEvents();
		}

		~Rendering() {
			glDeleteVertexArrays(1, &VAO);
			glDeleteBuffers(1, &VBO);

			glfwTerminate();
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

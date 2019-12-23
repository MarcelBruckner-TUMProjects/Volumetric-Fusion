#pragma once

#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace vc::rendering {
	class Shader
	{
	public:
		unsigned int ID;

		// activate the shader
		// ------------------------------------------------------------------------
		void use()
		{
			glUseProgram(ID);
		}
		// utility uniform functions
		// ------------------------------------------------------------------------
		void setBool(const std::string& name, bool value) const
		{
			glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
		}
		// ------------------------------------------------------------------------
		void setInt(const std::string& name, int value) const
		{
			glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
		}
		// ------------------------------------------------------------------------
		void setFloat(const std::string& name, float value) const
		{
			glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
		}
		// ------------------------------------------------------------------------
		void setVec2(const std::string& name, const float x, const float y) const
		{
			glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
		}

		// ------------------------------------------------------------------------
		void setVec3(const std::string& name, const glm::vec3 vec) const
		{
			setVec3(name, vec.x, vec.y, vec.z);
		}

		// ------------------------------------------------------------------------
		void setVec3(const std::string& name, const float x, const float y, const float z) const
		{
			glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
		}
		// ------------------------------------------------------------------------
		void setColor(const std::string& name, float r, float g, float b, float a) const
		{
			glUniform4f(glGetUniformLocation(ID, name.c_str()), r, g, b, a);
		}

		void setMat3(const std::string& name, const glm::mat3 matrix) {
			glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(matrix));
		}
		void setMat4(const std::string& name, const glm::mat4 matrix) {
			glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(matrix));
		}

		void setArray3(const std::string& name, GLfloat* vec, int count) {
			glUniform3fv(glGetUniformLocation(ID, name.c_str()), count, vec);
		}

	protected:
		// utility function for checking shader compilation/linking errors.
		// ------------------------------------------------------------------------
		void checkCompileErrors(unsigned int shader, std::string type)
		{
			int success;
			char infoLog[1024];
			if (type != "PROGRAM")
			{
				glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
				if (!success)
				{
					glGetShaderInfoLog(shader, 1024, NULL, infoLog);
					std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
				}
			}
			else
			{
				glGetProgramiv(shader, GL_LINK_STATUS, &success);
				if (!success)
				{
					glGetProgramInfoLog(shader, 1024, NULL, infoLog);
					std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
				}
			}
		}
	};

	class VertexFragmentShader : public Shader {
	public :
		VertexFragmentShader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr) : Shader()
		{
			// 1. retrieve the vertex/fragment source code from filePath
			std::string vertexCode;
			std::string fragmentCode;
			std::string geometryCode;
			std::ifstream vShaderFile;
			std::ifstream fShaderFile;
			std::ifstream gShaderFile;
			// ensure ifstream objects can throw exceptions:
			vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			try
			{
				// open files
				vShaderFile.open(vertexPath);
				std::stringstream vShaderStream;
				vShaderStream << vShaderFile.rdbuf();
				vShaderFile.close();
				vertexCode = vShaderStream.str();

				fShaderFile.open(fragmentPath);
				std::stringstream fShaderStream;
				fShaderStream << fShaderFile.rdbuf();
				fShaderFile.close();
				fragmentCode = fShaderStream.str();

				if (geometryPath != nullptr) {
					gShaderFile.open(geometryPath);
					std::stringstream gShaderStream;
					gShaderStream << gShaderFile.rdbuf();
					gShaderFile.close();
					geometryCode = gShaderStream.str();
				}
			}
			catch (std::ifstream::failure e)
			{
				std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
			}

			// vertex shader
			const char* vShaderCode = vertexCode.c_str();
			GLuint vertex;
			vertex = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertex, 1, &vShaderCode, NULL);
			glCompileShader(vertex);
			checkCompileErrors(vertex, "VERTEX");

			// fragment Shader
			const char* fShaderCode = fragmentCode.c_str();
			GLuint fragment;
			fragment = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragment, 1, &fShaderCode, NULL);
			glCompileShader(fragment);
			checkCompileErrors(fragment, "FRAGMENT");

			// geometry shader
			GLuint geometry;
			if (geometryPath != nullptr) {
				const char* gShaderCode = geometryCode.c_str();
				geometry = glCreateShader(GL_GEOMETRY_SHADER);
				glShaderSource(geometry, 1, &gShaderCode, NULL);
				glCompileShader(geometry);
				checkCompileErrors(geometry, "GEOMETRY");
			}

			// shader Program
			ID = glCreateProgram();
			glAttachShader(ID, vertex);
			glAttachShader(ID, fragment);
			if (geometryPath != nullptr) {
				glAttachShader(ID, geometry);
			}
			glLinkProgram(ID);
			checkCompileErrors(ID, "PROGRAM");
			// delete the shaders as they're linked into our program now and no longer necessary
			glDeleteShader(vertex);
			glDeleteShader(fragment);
			if (geometryPath != nullptr) {
				glDeleteShader(geometry);
			}
		}
	};

	class ComputeShader : Shader
	{
	public:
		ComputeShader(const char* computeShaderPath) : Shader()
		{
			// 1. retrieve the vertex/fragment source code from filePath
			std::string computeShaderCode;
			std::ifstream ComputeShaderFile;
			// ensure ifstream objects can throw exceptions:
			ComputeShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			try
			{
				// open files
				ComputeShaderFile.open(computeShaderPath);
				std::stringstream vShaderStream;
				// read file's buffer contents into streams
				vShaderStream << ComputeShaderFile.rdbuf();
				// close file handlers
				ComputeShaderFile.close();
				// convert stream into string
				computeShaderCode = vShaderStream.str();
			}
			catch (std::ifstream::failure e)
			{
				std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
			}
			const char* cShaderCode = computeShaderCode.c_str();
			// 2. compile shaders
			unsigned int computeShader;
			// vertex shader
			computeShader = glCreateShader(GL_COMPUTE_SHADER);
			glShaderSource(computeShader, 1, &cShaderCode, NULL);
			glCompileShader(computeShader);
			checkCompileErrors(computeShader, "COMPUTE");

			// shader Program
			ID = glCreateProgram();
			glAttachShader(ID, computeShader);
			glLinkProgram(ID);
			checkCompileErrors(ID, "COMPUTE PROGRAM");
			// delete the shaders as they're linked into our program now and no longer necessery
			glDeleteShader(computeShader);

			int work_grp_size[3];

			glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
			glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
			glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);

			printf("max local (in one shader) work group sizes x:%i y:%i z:%i\n",
				work_grp_size[0], work_grp_size[1], work_grp_size[2]);

			int work_grp_inv;
			glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);
			printf("max local work group invocations %i\n", work_grp_inv);
		}
	};
}
#endif
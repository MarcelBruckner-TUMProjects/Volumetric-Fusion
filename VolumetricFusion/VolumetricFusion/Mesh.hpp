#ifndef MESH_H
#define MESH_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

#include <VolumetricFusion\shader.hpp>

namespace vc::fusion{

	struct Vertex {
		glm::vec3 Position;
	};

	class Mesh {
		public:
			/*  Mesh Data  */
			std::vector<Vertex> vertices;
			std::vector<unsigned int> indices;

			vc::rendering::Shader* meshShader;

			/*  Functions  */
			Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices) {
				this->vertices = vertices;
				this->indices = indices;
				setupMesh();
			}

			void renderMesh(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {

				meshShader->use();
				//meshShader->setVec3("size", size);
				meshShader->setMat4("model", model);
				meshShader->setMat4("view", view);
				meshShader->setMat4("projection", projection);

				// draw mesh
				glBindVertexArray(VAO);
				glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
				glBindVertexArray(0);
			}

		private:
			/*  Render data  */
			unsigned int VAO, VBO, EBO;
			/*  Functions    */
			void setupMesh() {

				meshShader = new vc::rendering::VertexFragmentShader("shader/mesh.vs", "shader/mesh.fs");

				// create buffers/arrays
				glGenVertexArrays(1, &VAO);
				glGenBuffers(1, &VBO);
				glGenBuffers(1, &EBO);

				glBindVertexArray(VAO);
				// load data into vertex buffers
				glBindBuffer(GL_ARRAY_BUFFER, VBO);
				// A great thing about structs is that their memory layout is sequential for all its items.
				// The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
				// again translates to 3/2 floats which translates to a byte array.
				glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

				// set the vertex attribute pointers
				// vertex Positions
				glEnableVertexAttribArray(0);
				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

				glBindVertexArray(0);
			}
	};
}

#endif
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <VolumetricFusion\shader.hpp>

namespace vc::fusion {
	class Voxelgrid {
	private:
		float resolution;
		glm::vec3 radius;
		glm::vec3 origin;

		std::vector<float> points;
		unsigned int VBO, VAO;
		vc::rendering::Shader* shader;

	public:
		Voxelgrid(const float resolution = 0.1, const glm::vec3 radius = glm::vec3(2.0f), const glm::vec3 origin = glm::vec3(0.0f)) {
			this->resolution = resolution;
			this->origin = origin;
			this->radius = radius;

			for (float i = -radius.z; i <= radius.z; i += resolution)
			{
				for (float j = -radius.y; j <= radius.y; j += resolution)
				{
					for (float k = -radius.x; k <= radius.x; k += resolution)
					{
						points.push_back(k + origin.x);
						points.push_back(j + origin.y);
						points.push_back(i + origin.z);
					}
				}
			}
			initializeOpenGL();
		}

		void initializeOpenGL() {
			shader = new vc::rendering::Shader("shader/voxelgrid.vs", "shader/voxelgrid.fs");
			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
		}

		void render(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
			shader->use();
			shader->setVec3("radius", radius);
			shader->setMat4("model", model);
			shader->setMat4("view", view);
			shader->setMat4("projection", projection);
			glBindVertexArray(VAO);
			glDrawArrays(GL_POINTS, 0, points.size());
			glBindVertexArray(0);
		}
	};
}
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <VolumetricFusion\shader.hpp>
#include <unordered_map>

namespace vc::fusion {
	class Voxelgrid {
	private:
		float resolution;
		glm::vec3 size;
		glm::vec3 origin;

		std::vector<float> points;
		unsigned int VBO, VAO;
		vc::rendering::Shader* shader;

		std::unordered_map<float, float> tsdf;
		std::unordered_map<float, float> weights;

		glm::vec3 totalMin = glm::vec3((float)INT_MAX);
		glm::vec3 totalMax = glm::vec3((float)INT_MIN);

	public:
		Voxelgrid(const float resolution = 0.1, const glm::vec3 size = glm::vec3(5.0f), const glm::vec3 origin = glm::vec3(0.0f)) {
			this->resolution = resolution;
			this->origin = origin;
			this->size = size;
			
			for (float i = -size.x / 2.0f; i <= size.x / 2.0f; i += resolution)
			{
				for (float j = -size.y / 2.0f; j <= size.y / 2.0f; j += resolution)
				{
					for (float k = -size.z / 2.0f; k <= size.z / 2.0f; k += resolution)
					{
						points.push_back(i + origin.x);
						points.push_back(j + origin.y);
						points.push_back(k + origin.z);
						//voxels[hashFunc(i + origin.x, j + origin.y, k + origin.z)] = 7;
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
			shader->setVec3("size", size);
			shader->setMat4("model", model);
			shader->setMat4("view", view);
			shader->setMat4("projection", projection);
			glBindVertexArray(VAO);
			glDrawArrays(GL_POINTS, 0, points.size());
			glBindVertexArray(0);
		}

		float hashFunc(float x, float y, float z) {
			auto hash = x * size.y * size.z + y * size.z + z;
			std::cout << x << "," << y << "," << z << ": " << hash << std::endl;
			return hash;
		};


		void integrateFrame(const rs2::points points, glm::mat4 relativeTransformation) {

			const float* vertices_f = reinterpret_cast<const float*>(points.get_vertices());

			// insert them into the voxel grid (point by point)
			// yes, it is fucking slow
			
			auto size_half = size / 2.0f;

			for (int i = 0; i < points.size(); ++i) {
				// apply transformation
				auto index = i * 3;
				glm::vec3 vertex = glm::make_vec3(vertices_f + index);
				auto v = glm::vec4(vertex, 1.0);
				auto transformedVertex = relativeTransformation * v;

				int pt_grid_x = roundf(transformedVertex.x * 100 + size_half.x); //% voxel_size; // to cm
				int pt_grid_y = roundf(transformedVertex.y * 100 + size_half.y);
				int pt_grid_z = roundf(transformedVertex.z * 100 + size_half.z);

				// Convert voxel center from grid coordinates to base frame camera coordinates
				float pt_base_x = origin.x + pt_grid_x * resolution;
				float pt_base_y = origin.y + pt_grid_y * resolution;
				float pt_base_z = origin.z + pt_grid_z * resolution;

				int volume_idx = pt_grid_z * size.y * size.x + pt_grid_y * size.x + pt_grid_x;

				if (volume_idx >= size.x * size.y * size.z) {
					std::cout << "ERROR: volume_idx out of range" << std::endl;
					continue;
				}
				//float dist = fmin(1.0f, diff / trunc_margin);
				float weight_old = weights[volume_idx];
				float weight_new = weight_old + 1.0f;
				weights[volume_idx] = weight_new;
				//voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
				tsdf[volume_idx] = (tsdf[volume_idx] * weight_old) / weight_new;

				totalMin[0] = MIN(totalMin[0], transformedVertex.x);
				totalMax[0] = MAX(totalMax[0], transformedVertex.x);
				totalMin[1] = MIN(totalMin[1], transformedVertex.y);
				totalMax[1] = MAX(totalMax[1], transformedVertex.y);
				totalMin[2] = MIN(totalMin[2], transformedVertex.z);
				totalMax[2] = MAX(totalMax[2], transformedVertex.z);

				//std::cout << "(" << transformedVertex.x << "," << transformedVertex.y << "," << transformedVertex.z << ")" << std::endl;
			}

			std::cout << std::fixed << "Min: (" << totalMin[0] << "," << totalMin[1] << "," << totalMin[2] << ")" << std::endl;
			std::cout << std::fixed << "Max: (" << totalMax[0] << "," << totalMax[1] << "," << totalMax[2] << ")" << std::endl;

			std::cout << "" << std::endl;
		}
	};
}
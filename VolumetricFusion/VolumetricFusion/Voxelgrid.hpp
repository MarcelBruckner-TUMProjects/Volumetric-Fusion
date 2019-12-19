#ifndef _VOXELGRID_HEADER_
#define _VOXELGRID_HEADER_

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
		glm::vec3 sizeNormalized;
		glm::vec3 sizeHalf;
		glm::vec3 origin;

		std::vector<float> points;
		unsigned int VBO, VAO;
		vc::rendering::Shader* shader;

		std::vector<float> tsdf;
		std::vector<float> weights;

		glm::vec3 totalMin = glm::vec3((float)INT_MAX);
		glm::vec3 totalMax = glm::vec3((float)INT_MIN);

		std::map<int, std::vector<int>> integratedFramesPerPipeline;

		int num_gridPoints;

	public:
		Voxelgrid(const float resolution = 0.1, const glm::vec3 size = glm::vec3(5.0f), const glm::vec3 origin = glm::vec3(0.0f)) {
			this->resolution = resolution;
			this->origin = origin;
			this->size = size;
			this->sizeHalf = size / 2.0f;
			this->sizeNormalized = size / resolution;

			this->num_gridPoints = (sizeNormalized.x * sizeNormalized.y * sizeNormalized.z);

			tsdf = std::vector<float>(num_gridPoints);
			weights = std::vector<float>(num_gridPoints);

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
			shader = new vc::rendering::VertexFragmentShader("shader/voxelgrid.vs", "shader/voxelgrid.fs");
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

		int hashFunc(glm::vec3 v) {
			int hash = (int)v.z * (int)sizeNormalized.x * (int)sizeNormalized.y + (int)v.y * (int)sizeNormalized.x + (int)v.x;
			//std::cout << v.x << "," << v.y << "," << v.z << ": " << hash << std::endl;
			return hash;
		};

		glm::vec3 hashFuncInv(int hash) {
			int x = hash % (int)sizeNormalized.x;
			float y = (int)(hash / sizeNormalized.x) % (int)sizeNormalized.y;
			float z = (int)(hash / (sizeNormalized.x * size.y)) % (int)sizeNormalized.z;

			x /= resolution;
			y /= resolution;
			z /= resolution;

			std::cout << x << "," << y << "," << z << ": " << hash << std::endl;

			return glm::vec3(x, y, z);
		}


		void reset() {
			tsdf.clear();
			weights.clear();
			totalMin = glm::vec3((float)INT_MAX);
			totalMax = glm::vec3((float)INT_MIN);
		}

		void integrateFrameCPU(const rs2::points points, glm::mat4 relativeTransformation, const int pipelineId, const int frameId) {
			std::cout << "Integrating " << pipelineId << " - Frame: " << frameId << std::endl;
			
			if (integratedFramesPerPipeline.count(pipelineId) <= 0) {
				integratedFramesPerPipeline[pipelineId] = std::vector<int>();
			}
			else {
				if (std::find(integratedFramesPerPipeline[pipelineId].begin(), integratedFramesPerPipeline[pipelineId].end(), frameId) != integratedFramesPerPipeline[pipelineId].end()) {
					std::cout << "Already integrated." << std::endl << std::endl;
					return;
				}
			}
			integratedFramesPerPipeline[pipelineId].push_back(frameId);
			const rs2::vertex* vertices = points.get_vertices();

			// insert them into the voxel grid (point by point)
			// yes, it is fucking slow
			
			for (int i = 0; i < points.size(); ++i) {
				// apply transformation
				auto index = i;
				glm::vec4 vert = glm::vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1.0f);
				vert = relativeTransformation * vert;
				glm::vec3 v = glm::vec3(vert.x, vert.y, vert.z);

				//int pt_grid_x = roundf(transformedVertex.x * resolutionInv + sizeHalf.x); //% voxel_size; // to cm
				//int pt_grid_y = roundf(transformedVertex.y * resolutionInv + sizeHalf.y);
				//int pt_grid_z = roundf(transformedVertex.z * resolutionInv + sizeHalf.z);

				glm::vec3 pt_grid = (v + sizeHalf) / resolution;

				//// Convert voxel center from grid coordinates to base frame camera coordinates
				//float pt_base_x = origin.x + pt_grid_x * resolution;
				//float pt_base_y = origin.y + pt_grid_y * resolution;
				//float pt_base_z = origin.z + pt_grid_z * resolution;

				int volume_idx = hashFunc(pt_grid);

				if (volume_idx >= num_gridPoints || volume_idx < 0) {
					std::cout << "ERROR: (" << v.x << ", " << v.y << ", " << v.z << ")" << " not in grid!" << std::endl;
					continue;
				}
				//float dist = fmin(1.0f, diff / trunc_margin);
				float weight_old = weights[volume_idx];
				float weight_new = weight_old + 1.0f;
				weights[volume_idx] = weight_new;
				//voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
				tsdf[volume_idx] = (tsdf[volume_idx] * weight_old) / weight_new;

				totalMin.x = MIN(totalMin.x, v.x);
				totalMax.x = MAX(totalMax.x, v.x);
				totalMin.y = MIN(totalMin.y, v.y);
				totalMax.y = MAX(totalMax.y, v.y);
				totalMin.z = MIN(totalMin.z, v.z);
				totalMax.z = MAX(totalMax.z, v.z);

				//std::cout << "(" << transformedVertex.x << "," << transformedVertex.y << "," << transformedVertex.z << ")" << std::endl;
			}

			std::cout << std::fixed << "Min: (" << totalMin.x << "," << totalMin.y << "," << totalMin.z << ")" << std::endl;
			std::cout << std::fixed << "Max: (" << totalMax.x << "," << totalMax.y << "," << totalMax.z << ")" << std::endl;

			std::cout << "" << std::endl;
		}
	};
}

#endif
#ifndef _VOXELGRID_HEADER_
#define _VOXELGRID_HEADER_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <VolumetricFusion\shader.hpp>
#include <unordered_map>


namespace vc::fusion {
	const float cube_vertices[] = {
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 0.0f
	};
	const unsigned int cube_indices[] = {
		0, 1, 2,
		0, 2, 3,
		7, 6, 5,
		7, 5, 4,
		0, 4, 5,
		0, 5, 1,
		1, 5, 6,
		1, 6, 2,
		2, 6, 7,
		2, 7, 3,
		3, 7, 4,
		3, 4, 0,
	};

	class Voxelgrid {
	private:
		float resolution;
		glm::vec3 size;
		glm::vec3 sizeNormalized;
		glm::vec3 sizeHalf;
		glm::vec3 origin;

		std::vector<float> points;
		unsigned int VBO, VAO;
		vc::rendering::Shader* gridShader;

		GLuint VBO_cubes[4], VAO_cubes, EBO;
		vc::rendering::Shader* cubeShader;
		
		//const float voxel_size = 0.005f;
		//const float trunc_margin = voxel_size * 5;
	/*	float* voxel_grid_tsdf = nullptr;
		float* voxel_grid_weight = nullptr;*/

		float* tsdf;
		float* weights;
		int integratedFrames = 0;

		glm::vec3 totalMin = glm::vec3((float)INT_MAX);
		glm::vec3 totalMax = glm::vec3((float)INT_MIN);

		std::map<int, std::vector<int>> integratedFramesPerPipeline;

		int num_gridPoints;

	public:
		Voxelgrid(const float resolution = 1, const glm::vec3 size = glm::vec3(2.0f), const glm::vec3 origin = glm::vec3(0.0f)) {
			this->resolution = resolution;
			this->origin = origin;
			this->size = size;
			this->sizeHalf = size / 2.0f;
			this->sizeNormalized = size / resolution;
			this->sizeNormalized += glm::vec3(1.0f);

			this->num_gridPoints = (int)(sizeNormalized.x * sizeNormalized.y * sizeNormalized.z);
					   
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

			reset();

			initializeOpenGL();
		}

		void initializeOpenGL() {
			gridShader = new vc::rendering::VertexFragmentShader("shader/voxelgrid.vs", "shader/voxelgrid.fs");

			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			//cubeShader = new vc::rendering::VertexFragmentShader("shader/voxelgrid_cube.vs", "shader/voxelgrid_cube.fs");
			//glGenVertexArrays(1, &VAO_cubes);
			//glGenBuffers(4, VBO_cubes);

			//glBindVertexArray(VAO_cubes);

			//// the coordinates of the point grid never change
			//glBindBuffer(GL_ARRAY_BUFFER, VBO_cubes[0]);
			//glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);
			//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			//glEnableVertexAttribArray(0);

			//// same for the cube rendering object
			//glBindBuffer(GL_ARRAY_BUFFER, VBO);
			//glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
			//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			//glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
			//glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			//glEnableVertexAttribArray(3);

			//glBindVertexArray(0);

			//// Enable blending
			//glEnable(GL_BLEND);
			//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			//// Enable point size
			//glEnable(GL_PROGRAM_POINT_SIZE);
		}

		void renderGrid(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
			gridShader->use();
			gridShader->setVec3("size", size);
			gridShader->setMat4("model", model);
			gridShader->setMat4("view", view);
			gridShader->setMat4("projection", projection);
			glBindVertexArray(VAO);
			glDrawArrays(GL_POINTS, 0, (GLsizei)points.size());
			glBindVertexArray(0);
		}

		int hashFunc(glm::vec3 v) {
			int hash = (int)v.z * (int)sizeNormalized.x * (int)sizeNormalized.y + (int)v.y * (int)sizeNormalized.x + (int)v.x;
			//std::cout << v.x << "," << v.y << "," << v.z << ": " << hash << std::endl;
			return hash;
		}

		void renderField(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
			if (integratedFrames <= 0) {
				return;
			}

			cubeShader->use();

			glBindVertexArray(VAO_cubes);

			// bind the sdf values
			glBindBuffer(GL_ARRAY_BUFFER, VBO_cubes[1]);
			glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), tsdf, GL_STREAM_DRAW);
			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
			glEnableVertexAttribArray(1);

			// bind the weights
			glBindBuffer(GL_ARRAY_BUFFER, VBO_cubes[1]);
			glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), weights, GL_STREAM_DRAW);
			glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
			glEnableVertexAttribArray(2);

			cubeShader->setVec3("size", size);
			cubeShader->setMat4("model", model);
			cubeShader->setMat4("view", view);
			cubeShader->setMat4("projection", projection);
			
			glDrawArrays(GL_POINTS, 0, (GLsizei)points.size());

			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

			glBindVertexArray(0);
		}

		glm::vec3 hashFuncInv(int hash) {
			int x = hash % (int)sizeNormalized.x;
			int y = (int)(hash / sizeNormalized.x) % (int)sizeNormalized.y;
			int z = (int)(hash / (sizeNormalized.x * sizeNormalized.y)) % (int)sizeNormalized.z;

			glm::vec3 result = glm::vec3(x, y, z);
			result -= sizeHalf;
			
			std::cout << hash << ":\t" << result.x << "\t" << result.y << "\t" << result.z << std::endl;

			return result;
		}


		void reset() {
			delete tsdf;
			delete weights;

			tsdf = new float[num_gridPoints];
			weights = new float[num_gridPoints];
			/*if (voxel_grid_tsdf != nullptr) {
				delete[] voxel_grid_tsdf;
			}
			if (voxel_grid_weight != nullptr) {
				delete[] voxel_grid_weight;
			}
			int arraySize = roundf((size.x * size.y * size.z) * resolutionInv * resolutionInv * resolutionInv);
			std::cout << "Array size: " << arraySize << std::endl;
			voxel_grid_tsdf = new float[gridCount];
			voxel_grid_weight = new float[gridCount];
			memset(voxel_grid_tsdf, 0, arraySize * sizeof(float));
			memset(voxel_grid_weight, 0, arraySize * sizeof(float));*/

			integratedFrames = 0;

			totalMin = glm::vec3((float)INT_MAX);
			totalMax = glm::vec3((float)INT_MIN);
		}

		void integrateFrameCPU(const std::shared_ptr<vc::capture::CaptureDevice> pipeline, glm::mat4 relativeTransformation, const int pipelineId, const int frameId) {
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

			pipeline->depth_camera->K;

			integratedFramesPerPipeline[pipelineId].push_back(frameId);
			//const rs2::vertex* vertices = points.get_vertices();

			// insert them into the voxel grid (point by point)
			// yes, it is fucking slow
			std::stringstream ss;
			for (int i = 0; i < num_gridPoints; i++) {
				glm::vec3 pt_base = hashFuncInv(i);
				//std::cout << voxel.x << ", " << voxel.y << ", " << voxel.z << std::endl;

				pt_base += origin;

				//glm::vec4 vert = glm::vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1.0f);
				//vert = relativeTransformation * vert;
				//glm::vec3 v = glm::vec3(vert.x, vert.y, vert.z);

				//glm::vec3 pt_grid = (v + sizeHalf) / resolution;
				//
				//int volume_idx = hashFunc(pt_grid);

				//if (volume_idx >= num_gridPoints || volume_idx < 0) {
				//	ss << "ERROR: (" << v.x << ", " << v.y << ", " << v.z << ")" << " not in grid!" << std::endl;
				//	continue;
				//}
				////float dist = fmin(1.0f, diff / trunc_margin);
				//float weight_old = weights[volume_idx];
				//float weight_new = weight_old + 1.0f;
				//weights[volume_idx] = weight_new;
				////voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
				//float dist = 0;
				//tsdf[volume_idx] = ((tsdf[volume_idx] * weight_old) + dist) / weight_new;

				//totalMin.x = MIN(totalMin.x, v.x);
				//totalMax.x = MAX(totalMax.x, v.x);
				//totalMin.y = MIN(totalMin.y, v.y);
				//totalMax.y = MAX(totalMax.y, v.y);
				//totalMin.z = MIN(totalMin.z, v.z);
				//totalMax.z = MAX(totalMax.z, v.z);

				//std::cout << "(" << transformedVertex.x << "," << transformedVertex.y << "," << transformedVertex.z << ")" << std::endl;
			}


			std::cout << ss.str() << std::endl;

			std::cout << std::fixed << "Min: (" << totalMin.x << "," << totalMin.y << "," << totalMin.z << ")" << std::endl;
			std::cout << std::fixed << "Max: (" << totalMax.x << "," << totalMax.y << "," << totalMax.z << ")" << std::endl;
			integratedFrames++;

			std::cout << std::endl;
		}
	};
}

#endif
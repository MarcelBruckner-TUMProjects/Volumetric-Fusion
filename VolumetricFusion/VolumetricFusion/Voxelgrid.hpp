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
		float resolutionInv;
		glm::vec3 size;
		glm::vec3 dims;
		int gridCount;
		glm::vec3 origin;

		std::vector<float> points;
		unsigned int VBO, VAO;
		vc::rendering::Shader* gridShader;

		GLuint VBO_cubes[4], VAO_cubes, EBO;
		vc::rendering::Shader* cubeShader;

		std::unordered_map<int, float> tsdf;
		std::unordered_map<int, float> weights;

		//const float voxel_size = 0.005f;
		//const float trunc_margin = voxel_size * 5;
		float* voxel_grid_tsdf = nullptr;
		float* voxel_grid_weight = nullptr;

		int integratedFrames = 0;

		glm::vec3 totalMin = glm::vec3((float)INT_MAX);
		glm::vec3 totalMax = glm::vec3((float)INT_MIN);

		std::map<int, std::vector<int>> integratedFramesPerPipeline;

		std::ofstream outfile;

	public:
		Voxelgrid(const float resolution = 0.1, const glm::vec3 size = glm::vec3(5.0f), const glm::vec3 origin = glm::vec3(0.0f)) {
			this->resolution = resolution;
			this->resolutionInv = 1.0f / resolution;
			this->origin = origin;
			this->size = size;

			this->dims = glm::vec3(
				ceilf(size.x * resolutionInv),
				ceilf(size.y * resolutionInv),
				ceilf(size.z * resolutionInv)
			);
			this->gridCount = this->dims.x * this->dims.y * this->dims.z;

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

			outfile.open("scene.xyz", std::ios_base::app); // append instead of overwrite
		}

		void initializeOpenGL() {
			gridShader = new vc::rendering::Shader("shader/voxelgrid.vs", "shader/voxelgrid.fs");
			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			cubeShader = new vc::rendering::Shader("shader/voxelgrid_cube.vs", "shader/voxelgrid_cube.fs");
			glGenVertexArrays(1, &VAO_cubes);
			glGenBuffers(4, VBO_cubes);

			glBindVertexArray(VAO_cubes);

			// the coordinates of the point grid never change
			glBindBuffer(GL_ARRAY_BUFFER, VBO_cubes[0]);
			glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			// same for the cube rendering object
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(3);

			glBindVertexArray(0);

			// Enable blending
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			// Enable point size
			glEnable(GL_PROGRAM_POINT_SIZE);
		}

		void renderGrid(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
			gridShader->use();
			gridShader->setVec3("size", size);
			gridShader->setMat4("model", model);
			gridShader->setMat4("view", view);
			gridShader->setMat4("projection", projection);
			glBindVertexArray(VAO);
			glDrawArrays(GL_POINTS, 0, points.size());
			glBindVertexArray(0);
		}

		void renderField(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
			if (integratedFrames <= 0) {
				return;
			}

			cubeShader->use();

			glBindVertexArray(VAO_cubes);

			// bind the sdf values
			glBindBuffer(GL_ARRAY_BUFFER, VBO_cubes[1]);
			glBufferData(GL_ARRAY_BUFFER, gridCount * sizeof(float), voxel_grid_tsdf, GL_STREAM_DRAW);
			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
			glEnableVertexAttribArray(1);

			// bind the weights
			glBindBuffer(GL_ARRAY_BUFFER, VBO_cubes[1]);
			glBufferData(GL_ARRAY_BUFFER, gridCount * sizeof(float), voxel_grid_weight, GL_STREAM_DRAW);
			glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
			glEnableVertexAttribArray(2);

			cubeShader->setVec3("size", size);
			cubeShader->setMat4("model", model);
			cubeShader->setMat4("view", view);
			cubeShader->setMat4("projection", projection);
			
			glDrawArrays(GL_POINTS, 0, points.size());

			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

			glBindVertexArray(0);
		}

		float hashFunc(float x, float y, float z) {
			auto hash = x * size.y * size.z + y * size.z + z;
			std::cout << x << "," << y << "," << z << ": " << hash << std::endl;
			return hash;
		};


		void reset() {
			tsdf.clear();
			weights.clear();			

			if (voxel_grid_tsdf != nullptr) {
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
			memset(voxel_grid_weight, 0, arraySize * sizeof(float));

			integratedFrames = 0;

			totalMin = glm::vec3((float)INT_MAX);
			totalMax = glm::vec3((float)INT_MIN);
		}

		void integrateFrame(const rs2::points points, glm::mat4 relativeTransformation, const int pipelineId, const int frameId) {
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

				// check distance and out of bounds
				// transformed vertex shifted, normalize position to positive range in order to be able to calculate the index
				auto tvs = transformedVertex + glm::vec4(size_half.x, size_half.y, size_half.z, 1.0f);
				if (tvs.x < 0 || tvs.y < 0 || tvs.z < 0 || tvs.x > size.x || tvs.y > size.y || tvs.z > size.z) {
					continue;
				}

				//int pt_grid_x = roundf(transformedVertex.x * resolutionInv + size_half.x); //% voxel_size; // to cm
				//int pt_grid_y = roundf(transformedVertex.y * resolutionInv + size_half.y);
				//int pt_grid_z = roundf(transformedVertex.z * resolutionInv + size_half.z);

				int pt_grid_x = roundf(tvs.x * resolutionInv);
				int pt_grid_y = roundf(tvs.y * resolutionInv);
				int pt_grid_z = roundf(tvs.z * resolutionInv);

				// Same as hashFunc :)
				//int volume_idx = pt_grid_z * size.y * size.x + pt_grid_y * size.x + pt_grid_x;
				int volume_idx = pt_grid_z * dims.x * dims.y + pt_grid_y * dims.x + pt_grid_x;

				if (volume_idx < 0 || volume_idx >= gridCount) {
					//std::cout << "ERROR: volume_idx out of range (" << volume_idx << ")" << std::endl;
					continue;
				}

				//float weight_old = voxel_grid_weight[volume_idx];
				float dist = fmin(1.0f, sqrtf((v.x * v.x) + (v.y * v.y) + (v.z * v.z)));
				float weight_old = weights[volume_idx];
				float weight_new = weight_old + 1.0f;
				voxel_grid_weight[volume_idx] = weight_new;
				float tsdf_new = (voxel_grid_tsdf[volume_idx] * weight_old + dist) / weight_new;
				voxel_grid_tsdf[volume_idx] = 1.0f;
				//weights[volume_idx] = weight_new;
				//tsdf[volume_idx] = (tsdf[volume_idx] * weight_old + dist) / weight_new;

				totalMin[0] = MIN(totalMin[0], transformedVertex.x);
				totalMax[0] = MAX(totalMax[0], transformedVertex.x);
				totalMin[1] = MIN(totalMin[1], transformedVertex.y);
				totalMax[1] = MAX(totalMax[1], transformedVertex.y);
				totalMin[2] = MIN(totalMin[2], transformedVertex.z);
				totalMax[2] = MAX(totalMax[2], transformedVertex.z);

				// std::cout << "(" << transformedVertex.x << "," << transformedVertex.y << "," << transformedVertex.z << ")" << std::endl;

				outfile << transformedVertex.x << " " << transformedVertex.y << " " << transformedVertex.z << "\n";				
			}

			integratedFrames++;

			std::cout << std::fixed << "Min: (" << totalMin[0] << "," << totalMin[1] << "," << totalMin[2] << ")" << std::endl;
			std::cout << std::fixed << "Max: (" << totalMax[0] << "," << totalMax[1] << "," << totalMax[2] << ")" << std::endl;

			std::cout << "" << std::endl;
		}
	};
}

#endif
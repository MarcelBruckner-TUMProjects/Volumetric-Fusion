#ifndef _VOXELGRID_HEADER_
#define _VOXELGRID_HEADER_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <VolumetricFusion\shader.hpp>
#include <unordered_map>
#include "Utils.hpp"


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
		unsigned int VBOs[2], VAO;
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
		Voxelgrid(const float resolution = 0.05f, const glm::vec3 size = glm::vec3(2.0f), const glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f)) 
			: resolution(resolution), origin(origin), size(size), sizeHalf(size / 2.0f), sizeNormalized((size / resolution) + glm::vec3(1.0f)), num_gridPoints((sizeNormalized.x* sizeNormalized.y* sizeNormalized.z))
		{
			reset();

			int i = 0;
			for (int z = 0; z < sizeNormalized.z; z++)
			{
				for (int y = 0; y < sizeNormalized.y; y++)
				{
					for (int x = 0; x < sizeNormalized.x; x++)
					{
						std::stringstream ss;
						glm::vec3 voxelPosition = getVoxelPosition(x, y, z);

						int hash = z * sizeNormalized.y * sizeNormalized.x + y * sizeNormalized.x + x;

						//int hash = hashFunc(voxelPosition);
						//std::cout << hash << std::endl;

						tsdf[hash] = (1.0f * i++ / num_gridPoints) * 2.0f - 1.0f;

						ss << vc::utils::toString(&voxelPosition) << " (" << hash << ") --> " << tsdf[hash];
						//std::cout << ss.str() << std::endl;
						//continue;

						//i++;
						points.push_back(voxelPosition.x);
						points.push_back(voxelPosition.y);
						points.push_back(voxelPosition.z);
						//tsdf[hash] = 1.0f;

						//voxels[hashFunc(i + origin.x, j + origin.y, k + origin.z)] = 7;
					}
				}
			}

			//for (float z = -sizeHalf.z; z <= sizeHalf.z; z += resolution)
			//{
			//	for (float y = -sizeHalf.y; y <= sizeHalf.y; y += resolution)
			//	{
			//		for (float x = -sizeHalf.x; x <= sizeHalf.x; x += resolution)
			//		{
			//			std::stringstream ss;
			//			glm::vec3 voxelPosition = glm::vec3(x, y, z) + origin;

			//			int hash = hashFunc(voxelPosition);
			//			//std::cout << hash << std::endl;

			//			tsdf[hash] = (1.0f * i++ / num_gridPoints) * 2.0f - 1.0f;

			//			ss << vc::utils::toString(voxelPosition) << " (" << hash << ") --> " << tsdf[hash];
			//			std::cout << ss.str() << std::endl;
			//			//continue;

			//			//i++;
			//			points.push_back(x + origin.x);
			//			points.push_back(y + origin.y);
			//			points.push_back(z + origin.z);
			//			//tsdf[hash] = 1.0f;
			//			
			//			//voxels[hashFunc(i + origin.x, j + origin.y, k + origin.z)] = 7;
			//		}
			//	}
			//}


			initializeOpenGL();
		}

		void initializeOpenGL() {
			gridShader = new vc::rendering::VertexFragmentShader("shader/voxelgrid.vert", "shader/voxelgrid.frag");

			glGenVertexArrays(1, &VAO);
			glGenBuffers(2, VBOs);

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
			glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
			glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), tsdf, GL_STATIC_DRAW);

			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(1);

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

			//for (int i = 0; i < num_gridPoints; i++) {
			//	float value = tsdf[i];
			//	std::cout << value << std::endl;
			//}

			glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
			glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), tsdf, GL_STATIC_DRAW);

			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(1);

			glDrawArrays(GL_POINTS, 0, num_gridPoints);
			glBindVertexArray(0);
		}

		int hashFunc(glm::vec3 v) {
			glm::vec3 vv = v - origin;
			vv += sizeHalf;
			vv /= resolution;

			double hash = std::round(vv.z * sizeNormalized.x * sizeNormalized.y + vv.y * sizeNormalized.x + vv.x);
			//std::cout << vc::utils::toString(v) << ": " << hash << std::endl;
			return (int)hash;
		}

		//void renderField(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
		//	if (integratedFrames <= 0) {
		//		return;
		//	}

		//	cubeShader->use();

		//	glBindVertexArray(VAO_cubes);

		//	// bind the sdf values
		//	glBindBuffer(GL_ARRAY_BUFFER, VBO_cubes[1]);
		//	glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), tsdf, GL_STREAM_DRAW);
		//	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
		//	glEnableVertexAttribArray(1);

		//	// bind the weights
		//	glBindBuffer(GL_ARRAY_BUFFER, VBO_cubes[1]);
		//	glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), weights, GL_STREAM_DRAW);
		//	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
		//	glEnableVertexAttribArray(2);

		//	cubeShader->setVec3("size", size);
		//	cubeShader->setMat4("model", model);
		//	cubeShader->setMat4("view", view);
		//	cubeShader->setMat4("projection", projection);
		//	
		//	glDrawArrays(GL_POINTS, 0, points.size());

		//	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		//	//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		//	glBindVertexArray(0);
		//}

		glm::vec3 getVoxelPosition(int x, int y, int z) {
			glm::vec3 voxelPosition = glm::vec3(x, y, z);
			voxelPosition *= resolution;
			voxelPosition -= sizeHalf;
			voxelPosition += origin;
			return voxelPosition;
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

		void integrateFramesCPU(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines, std::vector<glm::mat4> relativeTransformations) {
			for (int i = 0; i < pipelines.size(); i++) {
				integrateFrameCPU(pipelines[i], relativeTransformations[i], i, pipelines[i]->data->frameId);
			}
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

			// insert them into the voxel grid (point by point)
			// yes, it is fucking slow

			glm::mat3 world2CameraProjection = pipeline->depth_camera->world2cam;
			rs2::depth_frame* depth_frame;
			int depth_width;
			int depth_height;

			try {
				depth_frame = (rs2::depth_frame*) & (pipeline->data->filteredDepthFrames);
				depth_width = depth_frame->as<rs2::video_frame>().get_width();
				depth_height = depth_frame->as<rs2::video_frame>().get_height();
			}
			catch (rs2::error & e) {
				return;
			}

			integratedFramesPerPipeline[pipelineId].push_back(frameId);

			int i = 0;
			for (int z = 0; z < sizeNormalized.z; z++)
			{
				for (int y = 0; y < sizeNormalized.y; y++)
				{
					for (int x = 0; x < sizeNormalized.x; x++)
					{
						std::stringstream ss;
						glm::vec3 voxelPosition = getVoxelPosition(x, y, z);

						int hash = z * sizeNormalized.y * sizeNormalized.x + y * sizeNormalized.x + x;

						//int hash = hashFunc(voxelPosition);
						//std::cout << hash << std::endl;

						tsdf[hash] = 1;

						//ss << vc::utils::toString(voxelPosition) << " (" << hash << ") --> " << tsdf[hash];
						//std::cout << ss.str() << std::endl;
						////continue;
						//continue;

						glm::vec3 projectedVoxelCenter = world2CameraProjection * voxelPosition;
						ss << " --> " << vc::utils::toString(&projectedVoxelCenter);

						float z = projectedVoxelCenter.z;

						if (z <= 0) {
							continue;
						}

						glm::vec2 pixelCoordinate = glm::vec2(projectedVoxelCenter.x, projectedVoxelCenter.y) / z;
						//pixelCoordinate /= 2.0f;
						ss << " --> " << vc::utils::toString(&pixelCoordinate) << " & " << z;

						if (pixelCoordinate.x < 0 || pixelCoordinate.y < 0 ||
							pixelCoordinate.x >= depth_width || pixelCoordinate.y >= depth_height) {
							continue;
						}

						ss << " <-- valid";

						float real_depth = depth_frame->get_distance(pixelCoordinate.x, pixelCoordinate.y);
						ss << " --- Voxel depth: " << z << " - Real depth: " << real_depth;

						float tsdf_value = z - real_depth;

						ss << " ==> TSDF value: " << tsdf_value;

						float clamped_tsdf_value = std::clamp(tsdf_value, -1.0f, 1.0f);

						ss << " (" << clamped_tsdf_value << ")";
						
						tsdf[hash] = clamped_tsdf_value;

						std::cout << ss.str() << std::endl;
					}
				}
			}
			std::cout << i << std::endl;

 			//for (int i = 0; i < num_gridPoints; i++) {
				//glm::vec3 pt_base = hashFuncInv(i);
				//std::cout << voxel.x << ", " << voxel.y << ", " << voxel.z << std::endl;

				//pt_base += origin;

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
			//}


			//std::cout << ss.str() << std::endl;

			/*std::cout << std::fixed << "Min: (" << totalMin.x << "," << totalMin.y << "," << totalMin.z << ")" << std::endl;
			std::cout << std::fixed << "Max: (" << totalMax.x << "," << totalMax.y << "," << totalMax.z << ")" << std::endl;
			integratedFrames++;

			std::cout << std::endl;*/
		}
	};
}
#endif
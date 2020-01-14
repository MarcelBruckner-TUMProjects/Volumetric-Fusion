#ifndef _VOXELGRID_HEADER_
#define _VOXELGRID_HEADER_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <VolumetricFusion\shader.hpp>
#include <unordered_map>
#include "Utils.hpp"
#include "Structs.hpp"

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
		unsigned int VBOs[4], VAO;
		vc::rendering::Shader* gridShader;

		GLuint VBO_cubes[4], VAO_cubes, EBO;
		vc::rendering::Shader* cubeShader;

		int integratedFrames = 0;

		glm::vec3 totalMin = glm::vec3((float)INT_MAX);
		glm::vec3 totalMax = glm::vec3((float)INT_MIN);

		std::map<int, std::vector<int>> integratedFramesPerPipeline;


	public:
		float resolution;
		glm::vec3 size;
		glm::vec3 sizeNormalized;
		glm::vec3 sizeHalf;
		glm::vec3 origin;

		float* tsdf;
		float* weights;
		float* isSet;

		int num_gridPoints;

		std::vector<float> points;

		int hashFunc(int x, int y, int z) {
			return z * sizeNormalized.y * sizeNormalized.x + y * sizeNormalized.x + x;
		}

		Voxelgrid(const float resolution = 0.5f, const glm::vec3 size = glm::vec3(2.0f), const glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f), bool initializeShader = true)
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

						int hash = this->hashFunc(x, y, z);

						tsdf[hash] = (1.0f * i++ / num_gridPoints) * 2.0f - 1.0f;

						points.push_back(voxelPosition.x);
						points.push_back(voxelPosition.y);
						points.push_back(voxelPosition.z);
					}
				}
			}

			if (initializeShader) {
				initializeOpenGL();
			}
		}

		void initializeOpenGL() {
			gridShader = new vc::rendering::VertexFragmentShader("shader/voxelgrid.vert", "shader/voxelgrid.frag");

			glGenVertexArrays(1, &VAO);
			glGenBuffers(4, VBOs);

			setTSDF();
		}

		void setTSDF() {
			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
			glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
			glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), tsdf, GL_STATIC_DRAW);

			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(1);

			glBindBuffer(GL_ARRAY_BUFFER, VBOs[2]);
			glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), weights, GL_STATIC_DRAW);

			glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(2);

			glBindBuffer(GL_ARRAY_BUFFER, VBOs[3]);
			glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), isSet, GL_STATIC_DRAW);

			glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(3);
		}

		void renderGrid(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
			gridShader->use();

			gridShader->setVec3("size", size);
			gridShader->setMat4("model", model);
			gridShader->setMat4("view", view);
			gridShader->setMat4("projection", projection);

			setTSDF();
			
			glDrawArrays(GL_POINTS, 0, num_gridPoints);
			glBindVertexArray(0);
		}

		glm::vec3 getVoxelPosition(int x, int y, int z) {
			glm::vec3 voxelPosition = glm::vec3(x, y, z);
			voxelPosition *= resolution;
			voxelPosition -= sizeHalf;
			voxelPosition += origin;
			return voxelPosition;
		}


		void reset() {
			delete[] tsdf;
			delete[] weights;
			delete[] isSet;

			tsdf = new float[num_gridPoints];
			weights = new float[num_gridPoints];
			isSet = new float[num_gridPoints];

			for (int i = 0; i < num_gridPoints; i++) {
				weights[i] = 0;
				tsdf[i] = 0;
				isSet[i] = 0;
			}

			integratedFrames = 0;

			totalMin = glm::vec3((float)INT_MAX);
			totalMax = glm::vec3((float)INT_MIN);
		}

		void integrateFramesCPU(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines, std::vector<Eigen::Matrix4d> relativeTransformations) {
			for (int i = 0; i < pipelines.size(); i++) {
				//continue;
				integrateFrameCPU(pipelines[i], relativeTransformations[i], i, pipelines[i]->data->frameId);
			}
		}

		void integrateFrameCPU(const std::shared_ptr<vc::capture::CaptureDevice> pipeline, Eigen::Matrix4d relativeTransformation, const int pipelineId, const int frameId) {
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

						int hash = this->hashFunc(x, y, z);

						//int hash = hashFunc(voxelPosition);
						//std::cout << hash << std::endl;

						//tsdf[hash] = 100;

						//ss << vc::utils::toString(voxelPosition) << " (" << hash << ") --> " << tsdf[hash];
						//std::cout << ss.str() << std::endl;
						////continue;
						//continue;

						glm::vec3 projectedVoxelCenter = world2CameraProjection * voxelPosition;
						//ss << " --> " << vc::utils::toString(&projectedVoxelCenter);

						float z = projectedVoxelCenter.z;

						if (z <= 0) {
							continue;
						}

						glm::vec2 pixelCoordinate = glm::vec2(projectedVoxelCenter.x, projectedVoxelCenter.y) / z;
						//pixelCoordinate /= 2.0f;
						//ss << " --> " << vc::utils::toString(&pixelCoordinate) << " & " << z;

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

						float old_tsdf = tsdf[hash];
						int old_weight = weights[hash];
						weights[hash] += 1;
						tsdf[hash] = (old_tsdf * old_weight + clamped_tsdf_value) / weights[hash];
						isSet[hash] = 7;

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

		vc::fusion::GridCell getGridCell(int x, int y, int z) {
			vc::fusion::GridCell cell;

			cell.p[0] = getVoxelPosition(x - 1, y - 1, z);
			cell.p[1] = getVoxelPosition(x, y - 1, z);
			cell.p[2] = getVoxelPosition(x, y - 1, z - 1);
			cell.p[3] = getVoxelPosition(x - 1, y - 1, z - 1);

			cell.p[4] = getVoxelPosition(x - 1, y, z);
			cell.p[5] = getVoxelPosition(x, y, z);
			cell.p[6] = getVoxelPosition(x, y, z - 1);
			cell.p[7] = getVoxelPosition(x - 1, y, z - 1);

			cell.val[0] = tsdf[hashFunc(x - 1, y - 1, z)];
			cell.val[1] = tsdf[hashFunc(x, y - 1, z)];
			cell.val[2] = tsdf[hashFunc(x, y - 1, z - 1)];
			cell.val[3] = tsdf[hashFunc(x - 1, y - 1, z - 1)];

			cell.val[4] = tsdf[hashFunc(x - 1, y, z)];
			cell.val[5] = tsdf[hashFunc(x, y, z)];
			cell.val[6] = tsdf[hashFunc(x, y, z - 1)];
			cell.val[7] = tsdf[hashFunc(x - 1, y, z - 1)];

			return cell;
		}

		glm::vec3* getVoxelCorners(int x, int y, int z) {
			glm::vec3* voxelCorners = new glm::vec3[8];

			for (int zz = 0; zz < 2; zz++)
			{
				for (int yy = 0; yy < 2; yy++)
				{
					for (int xx = 0; xx < 2; xx++)
					{
						voxelCorners[zz * 4 + yy * 2 + xx] = getVoxelPosition(x + xx, y + yy, z + zz);
					}
				}
			}
			return voxelCorners;
		}

		float* getTSDFValues(int x, int y, int z) {
			float* tsdfs = new float[8];

			for (int zz = 0; zz < 2; zz++)
			{
				for (int yy = 0; yy < 2; yy++)
				{
					for (int xx = 0; xx < 2; xx++)
					{
						tsdfs[zz * 4 + yy * 2 + xx] = tsdf[hashFunc(x + xx, y + yy, z + zz)];
					}
				}
			}
			return tsdfs;
		}
	};

	class SingleCellMockVoxelGrid : public Voxelgrid {
	public:
		SingleCellMockVoxelGrid() : Voxelgrid(1.0f, glm::vec3(1.0f), glm::vec3(0.0f, 0.0f, 0.0f), false)
		{
			tsdf[0] = 1.0f;
			tsdf[1] = 1.0f;
			tsdf[2] = 1.0f;
			tsdf[3] = -1.0f;
			tsdf[4] = 1.0f;
			tsdf[5] = 1.0f;
			tsdf[6] = 1.0f;
			tsdf[7] = 1.0f;
		}
	};

	class FourCellMockVoxelGrid : public Voxelgrid {
	public:
		FourCellMockVoxelGrid() : Voxelgrid(1.0f, glm::vec3(2.0f), glm::vec3(0.0f, 0.0f, 0.0f), false)
		{
			tsdf[0] = 1.0f;
			tsdf[1] = 1.0f;
			tsdf[2] = 1.0f;
			tsdf[3] = 1.0f;
			tsdf[4] = -1.0f;
			tsdf[5] = 1.0f;
			tsdf[6] = 1.0f;
			tsdf[7] = 1.0f;
			tsdf[8] = 1.0f;

			tsdf[0 + 9] = 1.0f;
			tsdf[1 + 9] = -1.0f;
			tsdf[2 + 9] = 1.0f;
			tsdf[3 + 9] = -1.0f;
			tsdf[4 + 9] = 1.0f;
			tsdf[5 + 9] = -1.0f;
			tsdf[6 + 9] = 1.0f;
			tsdf[7 + 9] = -1.0f;
			tsdf[8 + 9] = 1.0f;

			tsdf[0 + 9 + 9] = 1.0f;
			tsdf[1 + 9 + 9] = 1.0f;
			tsdf[2 + 9 + 9] = 1.0f;
			tsdf[3 + 9 + 9] = 1.0f;
			tsdf[4 + 9 + 9] = -1.0f;
			tsdf[5 + 9 + 9] = 1.0f;
			tsdf[6 + 9 + 9] = 1.0f;
			tsdf[7 + 9 + 9] = 1.0f;
			tsdf[8 + 9 + 9] = 1.0f;
		}
	};
}
#endif
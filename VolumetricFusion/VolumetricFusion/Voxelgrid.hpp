#pragma once

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
//#include "MarchingCubes.hpp"

namespace vc::fusion {
	const int INVALID_TSDF_VALUE = 5;

	   
	class Voxelgrid {
	protected:
		GLuint VAO;
		GLuint vbo;
		GLuint depthTexture;
		GLuint colorTexture;

		vc::rendering::Shader* gridShader;
		vc::rendering::Shader* tsdfComputeShader;
		vc::rendering::Shader* voxelgridComputeShader;

		int integratedFrames = 0;

		std::map<int, std::vector<int>> integratedFramesPerPipeline;
		float truncationDistance = 0.5f;

	public:
		float resolution;
		Eigen::Vector3d size;
		Eigen::Vector3i sizeNormalized;
		Eigen::Vector3d sizeHalf;
		Eigen::Vector3d origin;
		
		std::vector<Vertex> verts;

		//std::vector<float> tsdf;
		//std::vector<float> weights;

		int num_gridPoints;

		int hashFunc(int x, int y, int z) {
			//std::cout << z * sizeNormalized[1] * sizeNormalized[0] + y * sizeNormalized[0] + x << std::endl;
			return z * sizeNormalized[1] * sizeNormalized[0] + y * sizeNormalized[0] + x;
		}

		Voxelgrid(const float resolution = 0.01f, const Eigen::Vector3d size = Eigen::Vector3d(1.0, 1.0, 1.1), const Eigen::Vector3d origin = Eigen::Vector3d(0.0, 0.0, 1.0), bool initializeShader = true)
			: resolution(resolution), origin(origin), size(size), 
			sizeHalf(size / 2.0f), 
			sizeNormalized(Eigen::Vector3i((size / resolution).cast<int>()) + Eigen::Vector3i(1, 1, 1)), 
			num_gridPoints((sizeNormalized[0] * sizeNormalized[1] * sizeNormalized[2]))
		{
			//reset();

			verts = std::vector<Vertex>(num_gridPoints);
			
			if (initializeShader) {
				initializeOpenGL();
			}
		}

		void initializeOpenGL() {
			gridShader = new vc::rendering::VertexFragmentShader("shader/voxelgrid.vert", "shader/voxelgrid.frag", "shader/voxelgrid.geom");
			//tsdfComputeShader = new vc::rendering::ComputeShader("shader/tsdf.comp");
			voxelgridComputeShader = new vc::rendering::ComputeShader("shader/voxelgrid.comp");

			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &vbo);
			glGenTextures(1, &depthTexture);
			glGenTextures(1, &colorTexture);

			glBindTexture(GL_TEXTURE_2D, depthTexture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
				// set the texture wrapping parameters
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			// set texture filtering parameters
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			glBindTexture(GL_TEXTURE_2D, colorTexture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
				// set the texture wrapping parameters
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			// set texture filtering parameters
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			//setTSDF();

			resetVoxelgridBuffer();
		}

		void setComputeShader() {
			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * num_gridPoints, verts.data(), GL_DYNAMIC_COPY);

			glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0); // Vertex Attrib. 0
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)16); // Vertex Attrib. 1
			glEnableVertexAttribArray(1);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo);
		}

		void resetVoxelgridBuffer() {
			setComputeShader();

			voxelgridComputeShader->use();
			voxelgridComputeShader->setInt("INVALID_TSDF_VALUE", INVALID_TSDF_VALUE);
			voxelgridComputeShader->setFloat("resolution", resolution);
			voxelgridComputeShader->setVec3("sizeHalf", sizeHalf);
			voxelgridComputeShader->setVec3i("sizeNormalized", sizeNormalized);
			voxelgridComputeShader->setVec3("origin", origin);
			voxelgridComputeShader->setBool("setPosition", true);

			glDispatchCompute(num_gridPoints, 1, 1);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbo);
			glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Vertex) * num_gridPoints, verts.data());
		}
		
		void renderGrid(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {

			gridShader->use();

			gridShader->setFloat("cube_radius", resolution * 0.1f);
			gridShader->setVec3("size", size);
			gridShader->setMat4("model", model);
			gridShader->setMat4("view", view);
			gridShader->setMat4("projection", projection);
			gridShader->setMat4("coordinate_correction", vc::rendering::COORDINATE_CORRECTION);
			gridShader->setFloat("truncationDistance", truncationDistance);

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * num_gridPoints, verts.data(), GL_DYNAMIC_DRAW);

			glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0); // Vertex Attrib. 0
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)16); // Vertex Attrib. 1
			glEnableVertexAttribArray(1);

			glDrawArrays(GL_POINTS, 0, num_gridPoints);
			glBindVertexArray(0);
		}

		Eigen::Vector3d getVoxelPosition(int x, int y, int z) {
			Eigen::Vector3d voxelPosition = Eigen::Vector3d(x, y, z);
			voxelPosition *= resolution;
			voxelPosition -= sizeHalf;
			voxelPosition += origin;
			return voxelPosition;
		}


		//void reset() {
		//	//tsdf = std::vector<float>(num_gridPoints);
		//	//weights = std::vector<float>(num_gridPoints);
		//	//points = std::vector<float>(3 * num_gridPoints);

		//	integratedFrames = 0;
		//}

		void setTruncationDistance(float truncationDistance) {
			this->truncationDistance = truncationDistance;
		}

		virtual void integrateFrameGPU(const std::shared_ptr<vc::capture::CaptureDevice> pipeline, Eigen::Matrix4d relativeTransformation) try {
			glm::mat3 world2CameraProjection = pipeline->depth_camera->world2cam_glm;
			glm::mat3 colorWorld2CameraProjection = pipeline->rgb_camera->world2cam_glm;

			rs2::depth_frame depth_frame = pipeline->data->filteredDepthFrames;
			int depthWidth = depth_frame.as<rs2::video_frame>().get_width();
			int	depthHeight = depth_frame.as<rs2::video_frame>().get_height();

			rs2::frame color_frame = pipeline->data->filteredColorFrames;
			int colorWidth = color_frame.as<rs2::video_frame>().get_width();
			int	colorHeight = color_frame.as<rs2::video_frame>().get_height();

			setComputeShader();

			voxelgridComputeShader->use();
			voxelgridComputeShader->setInt("INVALID_TSDF_VALUE", INVALID_TSDF_VALUE);
			voxelgridComputeShader->setBool("setPosition", false);

			voxelgridComputeShader->setMat3("world2CameraProjection", world2CameraProjection);
			voxelgridComputeShader->setMat3("colorWorld2CameraProjection", colorWorld2CameraProjection);
			voxelgridComputeShader->setFloat("depthScale", pipeline->depth_camera->depthScale);
			voxelgridComputeShader->setVec2("depthResolution", depthWidth, depthHeight);
			voxelgridComputeShader->setVec2("colorResolution", colorWidth, colorHeight);
			voxelgridComputeShader->setFloat("truncationDistance", truncationDistance);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, depthTexture);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, depthWidth, depthHeight, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, depth_frame.get_data());
			voxelgridComputeShader->setInt("depthFrame", 0);

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, colorTexture);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, colorWidth, colorHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, color_frame.get_data());
			voxelgridComputeShader->setInt("colorFrame", 1);

			glDispatchCompute(num_gridPoints, 1, 1);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbo);
			glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Vertex) * num_gridPoints, verts.data());

			//for (int i = 0; i < num_gridPoints; i++) {
			//	if (std::abs(verts[i].pos[0]) < resolution * 0.9f && std::abs(verts[i].pos[1]) < resolution * 0.9f)
			//		//if (verts[i].pos[2] > 0 ) 
			//	{
			//		for (int j = 0; j < 4; j++) {
			//			std::cout <<
			//				verts[i].pos[j] << " | " << verts[i].tsdf[j] << std::endl;
			//		}
			//		std::cout << std::endl;
			//	}
			//}
			//std::cout << "";
		}
		catch (rs2::error & e) {
			return;
		}

		void integrateFramesCPU(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines, std::vector<Eigen::Matrix4d> relativeTransformations) {
			for (int i = 0; i < pipelines.size(); i++) {
				//continue;
				integrateFrameCPU(pipelines[i], relativeTransformations[i], i, pipelines[i]->data->frameId);
			}
		}

		void integrateFrameCPU(const std::shared_ptr<vc::capture::CaptureDevice> pipeline, Eigen::Matrix4d relativeTransformation, const int pipelineId, const int frameId) {
			std::cout << "Integrating " << pipelineId << " - Frame: " << frameId << std::endl;

			//if (integratedFramesPerPipeline.count(pipelineId) <= 0) {
			//	integratedFramesPerPipeline[pipelineId] = std::vector<int>();
			//}
			//else {
			//	if (std::find(integratedFramesPerPipeline[pipelineId].begin(), integratedFramesPerPipeline[pipelineId].end(), frameId) != integratedFramesPerPipeline[pipelineId].end()) {
			//		std::cout << "Already integrated." << std::endl << std::endl;
			//		return;
			//	}
			//}


			return;

			Eigen::Matrix3d world2CameraProjection = pipeline->depth_camera->world2cam;
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

			std::vector<std::thread> threads;
			for (int z = 0; z < sizeNormalized[2]; z++)
			{
				int yy = 0;
				for (int y = 0; y < sizeNormalized[1]; y++)
				{

					threads.emplace_back(std::thread([&, y, z]() {
						for (int x = 0; x < sizeNormalized[0]; x++)
						{
							GLfloat* tsdf;
							std::stringstream ss;
							Eigen::Vector3d voxelPosition = getVoxelPosition(x, y, z);

							int hash = this->hashFunc(x, y, z);

							ss << NAME_AND_VALUE(hash);
							ss << NAME_AND_VALUE(voxelPosition);

							Eigen::Vector3d projectedVoxelCenter = world2CameraProjection * voxelPosition;
							ss << NAME_AND_VALUE(projectedVoxelCenter);

							float z = projectedVoxelCenter[2];

							if (z <= 0) {
								tsdf = new GLfloat[4]{ (GLfloat)hash, INVALID_TSDF_VALUE,-3,-3 };
								ss << vc::utils::asHeader("Invalid because z <= 0");
							}
							else {
								Eigen::Vector2d pixelCoordinate = Eigen::Vector2d(projectedVoxelCenter[0], projectedVoxelCenter[1]) / z;
								ss << NAME_AND_VALUE(pixelCoordinate);

								//pixelCoordinate += Eigen::Vector2d(depth_width, depth_height);
								pixelCoordinate /= 2.0f;

								ss << NAME_AND_VALUE(pixelCoordinate);

								if (pixelCoordinate[0] < 0 || pixelCoordinate[1] < 0 ||
									pixelCoordinate[0] >= depth_width || pixelCoordinate[1] >= depth_height) {
									tsdf = new GLfloat[4]{ (GLfloat)hash, INVALID_TSDF_VALUE,-2,-2 };
									ss << vc::utils::asHeader("Invalid because pixel not in image");
								}
								else {
									try {
										float real_depth = depth_frame->get_distance(pixelCoordinate[0], pixelCoordinate[1]);
										ss << NAME_AND_VALUE(z);
										ss << NAME_AND_VALUE(real_depth);

										if (real_depth <= 0) {
											tsdf = new GLfloat[4]{ (GLfloat)hash, INVALID_TSDF_VALUE, -1, -1 };
											ss << vc::utils::asHeader("Invalid because no value in depth image");
										}
										else {
											float tsdf_value = real_depth - z;
											tsdf_value *= -1;

											ss << NAME_AND_VALUE(tsdf_value);

											float clamped_tsdf_value = std::clamp(tsdf_value, -1.0f, 1.0f);

											ss << NAME_AND_VALUE(clamped_tsdf_value);

											tsdf = new GLfloat[4]{ (GLfloat)hash, clamped_tsdf_value, 0, 0 };

											//float old_tsdf = tsdf[hash];
											//int old_weight = weights[hash];
											//weights[hash] += 1;
											//tsdf[hash] = (old_tsdf * old_weight + clamped_tsdf_value) / weights[hash];
											//isSet[hash] = 7;
										}
									}
									catch (rs2::error&) {
										std::cout << "error in retrieving depth" << std::endl;
									}
								}

							}

							verts[hash] = Vertex();
							//verts[hash].pos = new GLfloat[4]{ voxelPosition[0], voxelPosition[1], voxelPosition[2], 1.0f };
							//verts[hash].tsdf = tsdf;
						};
						//if (voxelPosition[0] == 0 && voxelPosition[1] == 0) {
						//	std::cout << ss.str();
						//	std::cout << std::endl;
						//}
					}));
					if (yy++ >= vc::utils::NUM_THREADS) {
						for (auto& thread : threads)
						{
							thread.join();
						}
						threads = std::vector<std::thread>();
						yy = 0;
					}
				}
				std::cout << "Calculated TSDF layer " << z << std::endl;
			}

			for (auto& thread : threads)
			{
				thread.join();
			}

			//vc::utils::sleepFor("", 1000);
		}

		bool getGridCell(int x, int y, int z, vc::fusion::GridCell* cell) {
			cell->verts[0] = verts[hashFunc(x, y, z + 1)];
			cell->verts[1] = verts[hashFunc(x + 1, y, z + 1)];
			cell->verts[2] = verts[hashFunc(x + 1, y, z)];
			cell->verts[3] = verts[hashFunc(x, y, z)];
								   
			cell->verts[4] = verts[hashFunc(x, y + 1, z + 1)];
			cell->verts[5] = verts[hashFunc(x + 1, y + 1, z + 1)];
			cell->verts[6] = verts[hashFunc(x + 1, y + 1, z)];
			cell->verts[7] = verts[hashFunc(x, y + 1, z)];

			return true;
		}
	};

	class SingleCellMockVoxelGrid : public Voxelgrid {
	public:
		SingleCellMockVoxelGrid() : Voxelgrid(1.0, Eigen::Vector3d(1.0, 1.0, 1.0), Eigen::Vector3d::Zero(), false)
		{
			initializeOpenGL();

			for (int i = 0; i < 8; i++)
			{
				verts[i].tsdf.y = 1.0f;
			}

			verts[0].tsdf.y = -1.0f;
		}
		
		void integrateFrameGPU(const std::shared_ptr<vc::capture::CaptureDevice> pipeline, Eigen::Matrix4d relativeTransformation, float truncationDistance) try {

		}
		catch (rs2::error & e) {
			return;
		}

	};

	class FourCellMockVoxelGrid : public Voxelgrid {
	public:
		FourCellMockVoxelGrid() : Voxelgrid(1.0, Eigen::Vector3d(2.0, 2.0, 2.0), Eigen::Vector3d::Identity(), false)
		{
			initializeOpenGL();


			for (int i = 0; i < 27; i++)
			{
				verts[i].tsdf.y = 0.5f;
				verts[i].color = glm::vec4(i % 3 == 0, (i + 1) % 3 == 0, (i + 2) % 3 == 0, 1);
			}

			//verts[0].tsdf.y = -0.5f;

			verts[4].tsdf.y = -0.5f;

			verts[1 + 9].tsdf.y = -0.5f;
			verts[3 + 9].tsdf.y = -0.5f;
			verts[5 + 9].tsdf.y = -0.5f;
			verts[7 + 9].tsdf.y = -0.5f;

			verts[4 + 9 + 9].tsdf.y = -0.5f;
			//verts[8 + 9 + 9].tsdf.y = -0.5f;
		}
		
		void integrateFrameGPU(const std::shared_ptr<vc::capture::CaptureDevice> pipeline, Eigen::Matrix4d relativeTransformation) try {
		}
		catch (rs2::error & e) {
			return;
		}
	};
}
#endif
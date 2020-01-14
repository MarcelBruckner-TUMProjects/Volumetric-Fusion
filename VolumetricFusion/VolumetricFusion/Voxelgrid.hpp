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
	const int INVALID_TSDF_VALUE = 5;

	class Voxelgrid {
	private:
		unsigned int VBOs[4], VAO;
		vc::rendering::Shader* gridShader;

		int integratedFrames = 0;

		std::map<int, std::vector<int>> integratedFramesPerPipeline;


	public:
		float resolution;
		Eigen::Vector3d size;
		Eigen::Vector3d sizeNormalized;
		Eigen::Vector3d sizeHalf;
		Eigen::Vector3d origin ;

		float* tsdf;
		float* weights;

		int num_gridPoints;

		std::vector<float> points;

		int hashFunc(int x, int y, int z) {
			return z * sizeNormalized[1] * sizeNormalized[0] + y * sizeNormalized[0] + x;
		}

		Voxelgrid(const float resolution = 1.f, const Eigen::Vector3d size = Eigen::Vector3d(2.0, 2.0, 2.0), const Eigen::Vector3d origin = Eigen::Vector3d(0.0, 0.0, 1.0), bool initializeShader = true)
			: resolution(resolution), origin(origin), size(size), sizeHalf(size / 2.0f), sizeNormalized((size / resolution) + Eigen::Vector3d(1.0, 1.0, 1.0)), num_gridPoints((sizeNormalized[0]* sizeNormalized[1]* sizeNormalized[2]))
		{
			reset();

			int i = 0;
			for (int z = 0; z < sizeNormalized[2]; z++)
			{
				for (int y = 0; y < sizeNormalized[1]; y++)
				{
					for (int x = 0; x < sizeNormalized[0]; x++)
					{
						std::stringstream ss;
						Eigen::Vector3d voxelPosition = getVoxelPosition(x, y, z);

						int hash = this->hashFunc(x, y, z);

						//tsdf[hash] = ((1.0f * i++) / num_gridPoints) * 2.0f - 1.0f;
						
						points.push_back(voxelPosition[0]);
						points.push_back(voxelPosition[1]);
						points.push_back(voxelPosition[2]);
					}
				}
			}

			if (initializeShader) {
				initializeOpenGL();
			}
		}

		void initializeOpenGL() {
			gridShader = new vc::rendering::VertexFragmentShader("shader/voxelgrid.vert", "shader/voxelgrid.frag", "shader/voxelgrid.geom");

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
		}

		void renderGrid(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
			gridShader->use();

			gridShader->setFloat("cube_radius", 0.05f);
			gridShader->setVec3("size", size);
			gridShader->setMat4("model", model);
			gridShader->setMat4("view", view);
			gridShader->setMat4("projection", projection);
			gridShader->setMat4("coordinate_correction", vc::rendering::COORDINATE_CORRECTION);

			setTSDF();
			
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


		void reset() {
			delete[] tsdf;
			delete[] weights;

			tsdf = new float[num_gridPoints];
			weights = new float[num_gridPoints];

			for (int i = 0; i < num_gridPoints; i++) {
				weights[i] = 0;
				tsdf[i] = INVALID_TSDF_VALUE;
			}

			integratedFrames = 0;
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

			for (int z = 0; z < sizeNormalized[2]; z++)
			{
				for (int y = 0; y < sizeNormalized[1]; y++)
				{
					for (int x = 0; x < sizeNormalized[0]; x++)
					{
						std::stringstream ss;
						Eigen::Vector3d voxelPosition = getVoxelPosition(x, y, z);

						int hash = this->hashFunc(x, y, z);

						ss << NAME_AND_VALUE(hash);
						ss << NAME_AND_VALUE(voxelPosition);

						Eigen::Vector3d projectedVoxelCenter = world2CameraProjection * voxelPosition;
						ss << NAME_AND_VALUE(projectedVoxelCenter);

						float z = projectedVoxelCenter[2];

						if (z <= 0) {
							tsdf[hash] = INVALID_TSDF_VALUE;
							ss << vc::utils::asHeader("Invalid because z <= 0");
						}
						else {
							Eigen::Vector2d pixelCoordinate = Eigen::Vector2d(projectedVoxelCenter[0], projectedVoxelCenter[1]) / z;
							ss << NAME_AND_VALUE(pixelCoordinate);

							//pixelCoordinate += Eigen::Vector2d(depth_width, depth_height);
							//pixelCoordinate /= 2.0f;

							//ss << NAME_AND_VALUE(pixelCoordinate);

							if (pixelCoordinate[0] < 0 || pixelCoordinate[1] < 0 ||
								pixelCoordinate[0] >= depth_width || pixelCoordinate[1] >= depth_height) {
								tsdf[hash] = INVALID_TSDF_VALUE;
								ss << vc::utils::asHeader("Invalid because pixel not in image");
							}
							else {
								float real_depth = depth_frame->get_distance(pixelCoordinate[0], pixelCoordinate[1]);
								ss << NAME_AND_VALUE(z);
								ss << NAME_AND_VALUE(real_depth);

								float tsdf_value = z - real_depth;

								ss << NAME_AND_VALUE(tsdf_value);
		
								float clamped_tsdf_value = std::clamp(tsdf_value, -1.0f, 1.0f);

								ss << NAME_AND_VALUE(clamped_tsdf_value);

								tsdf[hash] = clamped_tsdf_value;

								//float old_tsdf = tsdf[hash];
								//int old_weight = weights[hash];
								//weights[hash] += 1;
								//tsdf[hash] = (old_tsdf * old_weight + clamped_tsdf_value) / weights[hash];
								//isSet[hash] = 7;
							}
						}
						//std::cout << ss.str();
						//std::cout << std::endl;
					}
				}
			}
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

		Eigen::Vector3d* getVoxelCorners(int x, int y, int z) {
			Eigen::Vector3d* voxelCorners = new Eigen::Vector3d[8];

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
		SingleCellMockVoxelGrid() : Voxelgrid(1.0, Eigen::Vector3d(1.0, 1.0, 1.0), Eigen::Vector3d::Identity(), false)
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
		FourCellMockVoxelGrid() : Voxelgrid(1.0, Eigen::Vector3d(2.0, 2.0, 2.0), Eigen::Vector3d::Identity(), false)
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
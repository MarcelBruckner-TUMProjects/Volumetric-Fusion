#pragma once


#ifndef _FUSION_HEADER_
#define _FUSION_HEADER_


//#include <librealsense2/hpp/rs_processing.hpp>
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//
//#include "data.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace vc::fusion {
	class Fusion {
	private:
		//float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
		//float voxel_grid_origin_y = -1.5f;
		//float voxel_grid_origin_z = 0.5f;
		float voxel_grid_origin_x = 0.0f;
		float voxel_grid_origin_y = 0.0f;
		float voxel_grid_origin_z = 0.0f;
		float voxel_size = 1.0f; // in cm //0.006f;
		//float trunc_margin = voxel_size * 5;
		int voxel_grid_dim_x = 500; // in cm
		int voxel_grid_dim_y = 500;
		int voxel_grid_dim_z = 500;

		//float* voxel_grid_TSDF;
		//float* voxel_grid_weight;

		//float* gpu_voxel_grid_TSDF;
		//float* gpu_voxel_grid_weight;

		//float* gpu_cam_K;
		//float* gpu_cam2base;
		//float* gpu_depth_im;

		void initializeVoxelGrid() {
			//voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
			//voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];

			//for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i) {
			//	voxel_grid_TSDF[i] = 1.0f;
			//}
			//memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
		}

		void loadVariablesToGPUMemory() {
			//cudaMalloc(&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
			//cudaMalloc(&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
			//checkCUDA(__LINE__, cudaGetLastError());
			//cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
			//cudaMemcpy(gpu_voxel_grid_weight, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
			//checkCUDA(__LINE__, cudaGetLastError());

			//cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
			//cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
			//cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
			////cudaMalloc(&gpu_depth_im, im_height * im_width * sizeof(float));
			//checkCUDA(__LINE__, cudaGetLastError());
		}

		float* voxel_grid_TSDF;
		float* voxel_grid_weight;

		float totalMin[3];
		float totalMax[3];

	public:
		Fusion() {
			totalMin[0] = totalMin[1] = totalMin[2] = (float)INT_MAX;
			totalMax[0] = totalMax[1] = totalMax[2] = (float)INT_MIN;

			initializeVoxelGrid();
			loadVariablesToGPUMemory();

			voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
			voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
		}

		~Fusion() {
			delete voxel_grid_weight;
			delete voxel_grid_TSDF;
		}

		void integrateFrame(const rs2::points points, glm::mat4 relativeTransformation) {

			const float* vertices_f = reinterpret_cast<const float*>(points.get_vertices());

			// insert them into the voxel grid (point by point)
			// yes, it is fucking slow

			auto half_voxel_grid_dim_x = roundf(voxel_grid_dim_x / 2);
			auto half_voxel_grid_dim_y = roundf(voxel_grid_dim_y / 2);
			auto half_voxel_grid_dim_z = roundf(voxel_grid_dim_z / 2);

			for (int i = 0; i < points.size(); ++i) {
				// apply transformation
				auto index = i * 3;
				glm::vec3 vertex = glm::make_vec3(vertices_f + index);
				auto v = glm::vec4(vertex, 1.0);
				auto transformedVertex = relativeTransformation * v;

				int pt_grid_x = roundf(transformedVertex.x * 100 + half_voxel_grid_dim_x); //% voxel_size; // to cm
				int pt_grid_y = roundf(transformedVertex.y * 100 + half_voxel_grid_dim_y);
				int pt_grid_z = roundf(transformedVertex.z * 100 + half_voxel_grid_dim_z);

				// Convert voxel center from grid coordinates to base frame camera coordinates
				float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
				float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
				float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

				int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;

				if (volume_idx >= voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z) {
					std::cout << "ERROR: volume_idx out of range" << std::endl;
					continue;
				}
				//float dist = fmin(1.0f, diff / trunc_margin);
				float weight_old = voxel_grid_weight[volume_idx];
				float weight_new = weight_old + 1.0f;
				voxel_grid_weight[volume_idx] = weight_new;
				//voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
				voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old) / weight_new;

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
};

#endif // _FUSION_HEADER
#ifndef _FUSION_H_
#define _FUSION_H_

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <librealsense2/rs.hpp>

__global__
void Integrate(float* cam_K, float* cam2base, float* depth_im,
	int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
	float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
	float* voxel_grid_TSDF, float* voxel_grid_weight);

void tsdf_fusion(int pos_x, int pos_y, glm::mat3 intrinsics, glm::mat4 base2World, std::vector<rs2::points> pts);

#endif
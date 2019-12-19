//#include "fusion.cuh"

#include "fusion.cuh"
#include "utils.hpp"
#include < stdio.h >

using namespace std;

// CUDA kernel function to integrate a TSDF voxel volume given depth images
__global__
void Integrate(float* cam_K, float* cam2base, float* gpu_pts,
	int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
	float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
	float* voxel_grid_TSDF, float* voxel_grid_weight) {

	int pt_grid_z = blockIdx.x;
	int pt_grid_y = threadIdx.x;

	for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

		// Convert voxel center from grid coordinates to base frame camera coordinates
		float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
		float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
		float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

		// Convert from base frame camera coordinates to current frame camera coordinates
		float tmp_pt[3] = { 0 };
		tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
		tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
		tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
		float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
		float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
		float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

		if (pt_cam_z <= 0)
			continue;

		int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
		int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
		if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
			continue;

		//float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

		//printf("%d %d", pt_pix_y, pt_pix_x);

		//if (depth_val <= 0 || depth_val > 6)
		//	continue;

		//float diff = depth_val - pt_cam_z;

		//if (diff <= -trunc_margin)
		//	continue;

		//// Integrate
		//const int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
		//float dist = fmin(1.0f, diff / trunc_margin);
		//float weight_old = voxel_grid_weight[volume_idx];
		//float weight_new = weight_old + 1.0f;
		//voxel_grid_weight[volume_idx] = weight_new;
		//voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
	}
}

// Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
// Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
void tsdf_fusion(int pos_x, int pos_y, glm::mat3 intrinsics, std::map<int, glm::mat4> relativeTransformations, std::vector<rs2::points> pts) {
	
	//cout << vertices.size() << endl;

	int total_size = 0;

	for (int i = 0; i < pts.size(); i++) {
		total_size += pts[i].size();
	}
	
	cout << total_size << endl;
	cout << pts.size() << endl;

	cout << pts[0].size() << endl;
	cout << pts[1].size() << endl;
	cout << pts[2].size() << endl;
	cout << pts[3].size() << endl;

	float* pts3 = new float[(float)total_size * 3];

	for (int i = 0; i < pts.size(); i++) {

		if (pts[i].size() == 0)
			continue;
		
		const auto verts = pts[i].get_vertices();

		for (size_t j = 0; j < pts[i].size(); j++) {

			glm::vec4 res = relativeTransformations[i] * glm::vec4(verts[j].x, verts[j].y, verts[j].z, 1.0f);

			if (i == 0) {
				pts3[j * 3 + 0] = res[0];
				pts3[j * 3 + 1] = res[1];
				pts3[j * 3 + 2] = res[2];
			}
			else {
				pts3[pts[i - 1].size() + j * 3 + 0] = res[0];
				pts3[pts[i - 1].size() + j * 3 + 1] = res[1];
				pts3[pts[i - 1].size() + j * 3 + 2] = res[2];
			}

			cout << res[0] << " " << " " << res[1] << " " << res[2] << endl;
		}
	}


	// Location of folder containing RGB-D frames and camera pose files
	int base_frame_idx = 0;
	int first_frame_idx = 0;
	float num_frames = 1;

	float cam_K[3 * 3];
	float base2world[4 * 4];
	float cam2base[4 * 4];
	float cam2world[4 * 4];
	const int im_width = 1280;
	const int im_height = 720;
	float depth_im[im_height * im_width];

	// Voxel grid parameters (change these to change voxel grid resolution, etc.)
	float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
	float voxel_grid_origin_y = -1.5f;
	float voxel_grid_origin_z = 0.5f;
	float voxel_size = 0.006f;
	float trunc_margin = voxel_size * 5;
	int voxel_grid_dim_x = 500;
	int voxel_grid_dim_y = 500;
	int voxel_grid_dim_z = 500;
	
	const float* pSource = (const float*)glm::value_ptr(intrinsics);
	
	for (int i = 0; i < 9; i++)
		cam_K[i] = pSource[i];

	const float* pSource2 = (const float*)glm::value_ptr(relativeTransformations[0]);

	for (int i = 0; i < 16; i++)
		base2world[i] = pSource2[i];

	// Invert base frame camera pose to get world-to-base frame transform 
	float base2world_inv[16] = { 0 };
	invert_matrix(base2world, base2world_inv);

	// Initialize voxel grid
	float* voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
	float* voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
	for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
		voxel_grid_TSDF[i] = 1.0f;
	memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

	// Load variables to GPU memory
	float* gpu_voxel_grid_TSDF;
	float* gpu_voxel_grid_weight;
	cudaMalloc(&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
	cudaMalloc(&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
	checkCUDA(__LINE__, cudaGetLastError());
	cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_voxel_grid_weight, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDA(__LINE__, cudaGetLastError());
	float* gpu_cam_K;
	float* gpu_cam2base;
	float* gpu_pts;
	cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
	cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
	cudaMalloc(&gpu_pts, total_size * sizeof(float));
	checkCUDA(__LINE__, cudaGetLastError());

	// Loop through each depth frame and integrate TSDF voxel grid
	//first_frame_idx + 
	for (int frame_idx = first_frame_idx; frame_idx < (int)num_frames; ++frame_idx) {

		//depth_im
		//cam2world
		
		// Compute relative camera pose (camera-to-base frame)
		//multiply_matrix(base2world_inv, cam2world, cam2base);
		
		for(int i = 0; i < 16; i++)
			cam2base[i] = base2world[i];

		cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_pts, pts3, total_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDA(__LINE__, cudaGetLastError());

		//Integrate <<<voxel_grid_dim_z, voxel_grid_dim_y>>> (gpu_cam_K, gpu_cam2base, gpu_pts,
		//	im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
		//	voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
		//	gpu_voxel_grid_TSDF, gpu_voxel_grid_weight);
	}

	// Load TSDF voxel grid from GPU to CPU memory
	cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDA(__LINE__, cudaGetLastError());

	// Compute surface points from TSDF voxel grid and save to point cloud .ply file
	std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
	SaveVoxelGrid2SurfacePointCloud("tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
		voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
		voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f);

	//// Save TSDF voxel grid and its parameters to disk as binary file (float array)
	//std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
	//std::string voxel_grid_saveto_path = "tsdf.bin";
	//std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
	//float voxel_grid_dim_xf = (float)voxel_grid_dim_x;
	//float voxel_grid_dim_yf = (float)voxel_grid_dim_y;
	//float voxel_grid_dim_zf = (float)voxel_grid_dim_z;
	//std::cout << voxel_grid_dim_xf << " ";
	//outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
	//outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
	//outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
	//outFile.write((char*)&voxel_grid_origin_x, sizeof(float));
	//outFile.write((char*)&voxel_grid_origin_y, sizeof(float));
	//outFile.write((char*)&voxel_grid_origin_z, sizeof(float));
	//outFile.write((char*)&voxel_size, sizeof(float));
	//outFile.write((char*)&trunc_margin, sizeof(float));
	//for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
	//	outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
	//outFile.close();
}



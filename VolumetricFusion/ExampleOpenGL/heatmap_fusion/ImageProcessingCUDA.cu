#include "ImageProcessingCUDA.h"

#include <surface_functions.h>
#include <common_utils/Common.h>

#include "SVD3CUDA.cuh"

#define USE_NORMAL_PCA

namespace heatmap_fusion {
	namespace image_proc_gpu {

		/**
		 * Unprojects a 2D pixel, given its depth and inverse of camera pose and intrinsics.
		 * @returns	A 3D point
		 */
		CPU_AND_GPU inline __forceinline__ Vec3f unprojectPixel(const int x, const int y, const float depth, const Mat4f& inversePose, const Mat3f& inverseIntrinsics) {
			Vec3f v{ depth * x, depth * y, depth };
			return inversePose * (inverseIntrinsics * v);
		}

		__global__ void _unprojectDepthImage(
			cudaSurfaceObject_t depthImage,
			cudaSurfaceObject_t observationImage,
			int width, int height,
			const Mat3f inverseIntrinsics,
			const Mat4f inversePose
		) {
			int x = blockDim.x*blockIdx.x + threadIdx.x;
			int y = blockDim.y*blockIdx.y + threadIdx.y;

			if (x < width && y < height) {
				// Read input depth.
				float depth;
				surf2Dread(&depth, depthImage, x * sizeof(float), y);

				// Backproject pixel.
				Vec3f p(0, 0, 0);
				if (depth > 0) {
					p = unprojectPixel(x, y, depth, inversePose, inverseIntrinsics);
				}

				// Just for debugging.
				//p = Vec3f(float(x) / width, float(y) / height, (float(x) / width) * (float(y) / height));
				//p = Vec3f(depth / 5.f, depth / 5.f, depth / 5.f);

				// Write to output surface.
				surf2Dwrite(p.r(), observationImage, x * sizeof(float4), y);
			}
		}

		void unprojectDepthImage(
			Texture2D_32F& depthImage,
			Texture2D_RGBA32F& observationImage,
			int width, int height,
			const Mat3f& depthIntrinsics,
			const Mat4f& depthExtrinsics
		) {
			cudaArray_t depthImageArray, observationImageArray;
			cudaSurfaceObject_t depthImageSurface, observationImageSurface;

			// Map surface object to the depth image.
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, depthImage.getResourcePtr()));
			CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&depthImageArray, *depthImage.getResourcePtr(), 0, 0));

			cudaResourceDesc depthImageDesc;
			depthImageDesc.resType = cudaResourceTypeArray;
			depthImageDesc.res.array.array = depthImageArray;

			CUDA_SAFE_CALL(cudaCreateSurfaceObject(&depthImageSurface, &depthImageDesc));

			// Map surface object to the depth image.
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, observationImage.getResourcePtr()));
			CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&observationImageArray, *observationImage.getResourcePtr(), 0, 0));

			cudaResourceDesc observationImageDesc;
			observationImageDesc.resType = cudaResourceTypeArray;
			observationImageDesc.res.array.array = observationImageArray;

			CUDA_SAFE_CALL(cudaCreateSurfaceObject(&observationImageSurface, &observationImageDesc));

			// Execute depth unprojection.
			int blockDimX = 16;
			int blockDimY = 16;
			int gridDimX = (width + blockDimX - 1) / blockDimX;
			int gridDimY = (height + blockDimY - 1) / blockDimY;

			Mat3f depthIntrinsicsInv = depthIntrinsics.getInverse();
			Mat4f depthExtrinsicsInv = depthExtrinsics.getInverse();

			_unprojectDepthImage<<< dim3(gridDimX, gridDimY), dim3(blockDimX, blockDimY) >>>(
				depthImageSurface,
				observationImageSurface,
				width, height,
				depthIntrinsicsInv,
				depthExtrinsicsInv
			);
			CUDA_CHECK_ERROR();

			// Cleanup surface objects.
			CUDA_SAFE_CALL(cudaDestroySurfaceObject(depthImageSurface));
			CUDA_SAFE_CALL(cudaDestroySurfaceObject(observationImageSurface));
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, depthImage.getResourcePtr()));
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, observationImage.getResourcePtr()));
			//CUDA_SAFE_CALL(cudaStreamSynchronize(0)); // TODO: Is it necessary?
		}

		__global__ void _computeNormalImage(
			cudaSurfaceObject_t observationImage,
			cudaSurfaceObject_t normalImage,
			int width, int height,
			int kernelRadius,
			float maxDistanceSq
		) {
			int x = blockDim.x*blockIdx.x + threadIdx.x;
			int y = blockDim.y*blockIdx.y + threadIdx.y;

			if (x < width && y < height) {
				// Read center observation.
				float4 observationRaw;
				surf2Dread(&observationRaw, observationImage, x * sizeof(float4), y);

				Vec3f observation(observationRaw);
				if (observation.z() <= 0) {
					surf2Dwrite(make_float4(0, 0, 0, 0), normalImage, x * sizeof(float4), y);
					return;
				}

#				ifndef USE_NORMAL_PCA

				/// Computation using central differences.
				if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) {
					surf2Dwrite(make_float4(0, 0, 0, 0), normalImage, x * sizeof(float4), y);
					return;
				}

				float4 neighborLeftRaw, neighborRightRaw, neighborBottomRaw, neighborTopRaw;
				surf2Dread(&neighborLeftRaw, observationImage, (x - 1) * sizeof(float4), y);
				surf2Dread(&neighborRightRaw, observationImage, (x + 1) * sizeof(float4), y);
				surf2Dread(&neighborBottomRaw, observationImage, x * sizeof(float4), y - 1);
				surf2Dread(&neighborTopRaw, observationImage, x * sizeof(float4), y + 1);

				Vec3f neighborLeft(neighborLeftRaw);
				Vec3f neighborRight(neighborRightRaw);
				Vec3f neighborBottom(neighborBottomRaw);
				Vec3f neighborTop(neighborTopRaw);

				if (Vec3f::distSq(neighborLeft, observation) > maxDistanceSq ||
					Vec3f::distSq(neighborRight, observation) > maxDistanceSq ||
					Vec3f::distSq(neighborBottom, observation) > maxDistanceSq ||
					Vec3f::distSq(neighborTop, observation) > maxDistanceSq
					) {
					surf2Dwrite(make_float4(0, 0, 0, 0), normalImage, x * sizeof(float4), y);
				}
				else {
					Vec3f normal = -(neighborLeft - neighborRight) ^ (neighborBottom - neighborTop);
					normal.normalizeIfNonzero();
					surf2Dwrite(normal.r(), normalImage, x * sizeof(float4), y);
				}

#				else

				/// Computation using PCA.
				// Loop through all observations in the window and compute mean observation.
				int counter = 0;
				Vec3f center(0, 0, 0);

				for (int yi = y - kernelRadius; yi <= y + kernelRadius; ++yi) {
					for (int xi = x - kernelRadius; xi <= x + kernelRadius; ++xi) {
						if (xi >= 0 && xi < width && yi >= 0 && yi < height) {
							float4 pointNeighborRaw;
							surf2Dread(&pointNeighborRaw, observationImage, xi * sizeof(float4), yi);

							Vec3f pointNeighbor(pointNeighborRaw);
							if (pointNeighbor.z() > 0 && Vec3f::distSq(observation, pointNeighbor) <= maxDistanceSq) {
								center += pointNeighbor;
								++counter;
							}
						}
					}
				}

				center /= (float)counter;

				// Loop through all observations in the window and compute covariance matrix.
				float a11{ 0.f }, a12{ 0.f }, a13{ 0.f }, a21{ 0.f }, a22{ 0.f }, a23{ 0.f }, a31{ 0.f }, a32{ 0.f }, a33{ 0.f };

				for (int yi = y - kernelRadius; yi <= y + kernelRadius; ++yi) {
					for (int xi = x - kernelRadius; xi <= x + kernelRadius; ++xi) {
						if (xi >= 0 && xi < width && yi >= 0 && yi < height) {
							float4 pointNeighborRaw;
							surf2Dread(&pointNeighborRaw, observationImage, xi * sizeof(float4), yi);

							Vec3f pointNeighbor(pointNeighborRaw);
							if (pointNeighbor.z() > 0 && Vec3f::distSq(observation, pointNeighbor) <= maxDistanceSq) {
								Vec3f pointDiff = pointNeighbor - center;
								a11 += pointDiff.x() * pointDiff.x();
								a12 += pointDiff.x() * pointDiff.y();
								a13 += pointDiff.x() * pointDiff.z();
								a21 += pointDiff.y() * pointDiff.x();
								a22 += pointDiff.y() * pointDiff.y();
								a23 += pointDiff.y() * pointDiff.z();
								a31 += pointDiff.z() * pointDiff.x();
								a32 += pointDiff.z() * pointDiff.y();
								a33 += pointDiff.z() * pointDiff.z();
							}
						}
					}
				}

				// Compute SVD of covariance matrix to get the eigenvector with smallest eigenvalue.
				float u11, u12, u13, u21, u22, u23, u31, u32, u33;
				float s11, s12, s13, s21, s22, s23, s31, s32, s33;
				float v11, v12, v13, v21, v22, v23, v31, v32, v33;

				svd(a11, a12, a13, a21, a22, a23, a31, a32, a33,
					u11, u12, u13, u21, u22, u23, u31, u32, u33,
					s11, s12, s13, s21, s22, s23, s31, s32, s33,
					v11, v12, v13, v21, v22, v23, v31, v32, v33);

				Vec3f minEigenVector;
				if (s11 < s33 && s11 < s22) {
					minEigenVector = Vec3f(v11, v12, v13);
				}
				else if (s22 < s11 && s22 < s33) {
					minEigenVector = Vec3f(v21, v22, v23);
				}
				else {
					minEigenVector = Vec3f(v31, v32, v33);
				}

				// Write to output surface.
				if (minEigenVector.lengthSq() <= 0.f) {
					surf2Dwrite(make_float4(0, 0, 0, 0), normalImage, x * sizeof(float4), y);
					return;
				}

				if ((minEigenVector | observation) < 0) {
					surf2Dwrite(minEigenVector.getNormalized().r(), normalImage, x * sizeof(float4), y);
				}
				else {
					surf2Dwrite((-minEigenVector).getNormalized().r(), normalImage, x * sizeof(float4), y);
				}

#				endif
			}
		}

		void computeNormalImage(
			Texture2D_RGBA32F& observationImage, 
			Texture2D_RGBA32F& normalImage, 
			int width, int height,
			int kernelRadius, 
			float maxDistance
		) {
			cudaArray_t normalImageArray, observationImageArray;
			cudaSurfaceObject_t normalImageSurface, observationImageSurface;

			// Map surface object to the observation image.
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, observationImage.getResourcePtr()));
			CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&observationImageArray, *observationImage.getResourcePtr(), 0, 0));

			cudaResourceDesc observationImageDesc;
			observationImageDesc.resType = cudaResourceTypeArray;
			observationImageDesc.res.array.array = observationImageArray;

			CUDA_SAFE_CALL(cudaCreateSurfaceObject(&observationImageSurface, &observationImageDesc));
			
			// Map surface object to the normal image.
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, normalImage.getResourcePtr()));
			CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&normalImageArray, *normalImage.getResourcePtr(), 0, 0));

			cudaResourceDesc normalImageDesc;
			normalImageDesc.resType = cudaResourceTypeArray;
			normalImageDesc.res.array.array = normalImageArray;

			CUDA_SAFE_CALL(cudaCreateSurfaceObject(&normalImageSurface, &normalImageDesc));

			// Execute normal computation.
			int blockDimX = 16;
			int blockDimY = 16;
			int gridDimX = (width + blockDimX - 1) / blockDimX;
			int gridDimY = (height + blockDimY - 1) / blockDimY;

			float maxDistanceSq = maxDistance * maxDistance;

			_computeNormalImage <<< dim3(gridDimX, gridDimY), dim3(blockDimX, blockDimY) >>>(
				observationImageSurface,
				normalImageSurface,
				width, height,
				kernelRadius,
				maxDistanceSq
			);
			CUDA_CHECK_ERROR();

			// Cleanup surface objects.
			CUDA_SAFE_CALL(cudaDestroySurfaceObject(observationImageSurface));
			CUDA_SAFE_CALL(cudaDestroySurfaceObject(normalImageSurface));
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, observationImage.getResourcePtr()));
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, normalImage.getResourcePtr()));
		}

		__global__ void _filterInvalidPoints(
			cudaSurfaceObject_t observationImage,
			int width, int height,
			int* indexCounter,
			float4* validPoints
		) {
			int x = blockDim.x*blockIdx.x + threadIdx.x;
			int y = blockDim.y*blockIdx.y + threadIdx.y;

			if (x < width && y < height) {
				// Read center observation.
				float4 observationRaw;
				surf2Dread(&observationRaw, observationImage, x * sizeof(float4), y);

				Vec3f observation(observationRaw);
				if (observation.z() > 0) {
					// Get the unique correspondence index.
					int currentIdx = atomicAdd(indexCounter, 1);

					// Write the observation.
					validPoints[currentIdx] = observation.r();
				}
			}
		}

		void filterInvalidPoints(
			Texture2D_RGBA32F& observationImage,
			int width, int height,
			MemoryContainer<float4>& validPoints
		) {
			cudaArray_t observationImageArray;
			cudaSurfaceObject_t observationImageSurface;

			// Map surface object to the observation image.
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, observationImage.getResourcePtr()));
			CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&observationImageArray, *observationImage.getResourcePtr(), 0, 0));

			cudaResourceDesc observationImageDesc;
			observationImageDesc.resType = cudaResourceTypeArray;
			observationImageDesc.res.array.array = observationImageArray;

			CUDA_SAFE_CALL(cudaCreateSurfaceObject(&observationImageSurface, &observationImageDesc));

			// Allocate output array.
			MemoryContainer<float4> validPointsAll;
			validPointsAll.allocate(width * height, false, true);

			// Reserve the index counter.
			MemoryContainer<int> indexCounter;
			indexCounter.allocate(1, true, true);

			// Initialize index to 0.
			indexCounter.getElement(0, Type2Type<MemoryTypeCPU>()) = 0;
			indexCounter.copyHostToDevice();

			// Execute point filtering.
			int blockDimX = 16;
			int blockDimY = 16;
			int gridDimX = (width + blockDimX - 1) / blockDimX;
			int gridDimY = (height + blockDimY - 1) / blockDimY;

			_filterInvalidPoints<<< dim3(gridDimX, gridDimY), dim3(blockDimX, blockDimY) >>>(
				observationImageSurface,
				width, height,
				indexCounter.d(),
				validPointsAll.d()
			);
			CUDA_CHECK_ERROR();

			// Get the total number of valid observations by reading the final index value.
			indexCounter.copyDeviceToHost();
			unsigned nValidObservations = indexCounter.h(0);

			if (nValidObservations > 0) {
				// Copy valid observation data to output arrays.
				validPoints.clear();
				validPoints.allocate(nValidObservations, false, true);
				CUDA_SAFE_CALL(cudaMemcpy(validPoints.d(), validPointsAll.d(), nValidObservations * sizeof(float4), cudaMemcpyDeviceToDevice));

				validPoints.setUpdated(false, true);
			}
		}

	} // namespace image_proc_gpu
} // namespace heatmap_fusion
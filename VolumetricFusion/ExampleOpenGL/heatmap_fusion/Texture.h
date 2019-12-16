#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <common_utils/Common.h>
#include <common_utils/data_structures/Array2.h>

using namespace common_utils;

namespace heatmap_fusion {

	class Texture2D_32F {
	public:
		void create(Array2<float>& image);
		void create(int width, int height);

		//void execute() {
		//	cudaArray_t arrayObj;
		//	cudaSurfaceObject_t surfaceObj;

		//	// Map surface object to the resource.
		//	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_resource));
		//	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&arrayObj, m_resource, 0, 0));

		//	cudaResourceDesc wdsc;
		//	wdsc.resType = cudaResourceTypeArray;
		//	wdsc.res.array.array = arrayObj;

		//	CUDA_SAFE_CALL(cudaCreateSurfaceObject(&surfaceObj, &wdsc));
		//	
		//	//fillBlue(writeSurface, dim3(width, height));
		//	
		//	CUDA_SAFE_CALL(cudaDestroySurfaceObject(surfaceObj));
		//	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_resource));
		//	CUDA_SAFE_CALL(cudaStreamSynchronize(0)); // TODO: Is it necessary?
		//}

		GLuint getTexture() {
			return m_tex;
		}

		cudaGraphicsResource** getResourcePtr() {
			return &m_resource;
		}

	private:
		GLuint m_tex{ 0 };
		cudaGraphicsResource* m_resource{ nullptr };
	};



	class Texture2D_RGBA32F {
	public:
		void create(Array2<float4>& image);
		void create(int width, int height);

		//void execute() {
		//	cudaArray_t arrayObj;
		//	cudaSurfaceObject_t surfaceObj;

		//	// Map surface object to the resource.
		//	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_resource));
		//	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&arrayObj, m_resource, 0, 0));

		//	cudaResourceDesc wdsc;
		//	wdsc.resType = cudaResourceTypeArray;
		//	wdsc.res.array.array = arrayObj;

		//	CUDA_SAFE_CALL(cudaCreateSurfaceObject(&surfaceObj, &wdsc));
		//	
		//	//fillBlue(writeSurface, dim3(width, height));
		//	
		//	CUDA_SAFE_CALL(cudaDestroySurfaceObject(surfaceObj));
		//	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_resource));
		//	CUDA_SAFE_CALL(cudaStreamSynchronize(0)); // TODO: Is it necessary?
		//}

		GLuint getTexture() {
			return m_tex;
		}

		cudaGraphicsResource** getResourcePtr() {
			return &m_resource;
		}

	private:
		GLuint m_tex{ 0 };
		cudaGraphicsResource* m_resource{ nullptr };
	};


	class Texture2D_RGBA8UC {
	public:
		void create(Array2<uchar4>& image);
		void create(int width, int height);

		GLuint getTexture() {
			return m_tex;
		}

		cudaGraphicsResource** getResourcePtr() {
			return &m_resource;
		}

	private:
		GLuint m_tex{ 0 };
		cudaGraphicsResource* m_resource{ nullptr };
	};

} // namespace heatmap_fusion

#endif // !TEXTURE_H

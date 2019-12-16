#include "Texture.h"

namespace heatmap_fusion {

	void Texture2D_32F::create(Array2<float>& image) {
		int width = image.getDimX();
		int height = image.getDimY();

		// Create OpenGL texture and copy data to it.
		if (m_tex) glDeleteTextures(1, &m_tex);
		glGenTextures(1, &m_tex);
		glBindTexture(GL_TEXTURE_2D, m_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, image.getData());

		// Map CUDA resource to OpenGL texture.
		CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_resource, m_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	void Texture2D_32F::create(int width, int height) {
		// Create OpenGL texture and copy data to it.
		if (m_tex) glDeleteTextures(1, &m_tex);
		glGenTextures(1, &m_tex);
		glBindTexture(GL_TEXTURE_2D, m_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

		// Map CUDA resource to OpenGL texture.
		CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_resource, m_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	void Texture2D_RGBA32F::create(Array2<float4>& image) {
		int width = image.getDimX();
		int height = image.getDimY();

		// Create OpenGL texture and copy data to it.
		if (m_tex) glDeleteTextures(1, &m_tex);
		glGenTextures(1, &m_tex);
		glBindTexture(GL_TEXTURE_2D, m_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, image.getData());

		// Map CUDA resource to OpenGL texture.
		CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_resource, m_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	void Texture2D_RGBA32F::create(int width, int height) {
		// Create OpenGL texture and copy data to it.
		if (m_tex) glDeleteTextures(1, &m_tex);
		glGenTextures(1, &m_tex);
		glBindTexture(GL_TEXTURE_2D, m_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

		// Map CUDA resource to OpenGL texture.
		CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_resource, m_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	void Texture2D_RGBA8UC::create(Array2<uchar4>& image) {
		int width = image.getDimX();
		int height = image.getDimY();

		// Create OpenGL texture and copy data to it.
		if (m_tex) glDeleteTextures(1, &m_tex);
		glGenTextures(1, &m_tex);
		glBindTexture(GL_TEXTURE_2D, m_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.getData());

		// Map CUDA resource to OpenGL texture.
		CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_resource, m_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	void Texture2D_RGBA8UC::create(int width, int height) {
		// Create OpenGL texture and copy data to it.
		if (m_tex) glDeleteTextures(1, &m_tex);
		glGenTextures(1, &m_tex);
		glBindTexture(GL_TEXTURE_2D, m_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

		// Map CUDA resource to OpenGL texture.
		CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_resource, m_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

} // namespace heatmap_fusion
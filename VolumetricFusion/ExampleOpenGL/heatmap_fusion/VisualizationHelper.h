#ifndef VISUALIZATIONHELPER_H
#define VISUALIZATIONHELPER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <common_utils/memory_managment/MemoryContainer.h>
#include <common_utils/data_structures/Grid2Interface.h>
#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

using namespace common_utils;
using namespace matrix_lib;

namespace heatmap_fusion {

	class MeshBuffers {
	public:
		void create(MemoryContainer<float4>& positions, MemoryContainer<float4>& normals);
		void bind(int attributeOffset = 0);

		unsigned getNumVertices() { return m_nVertices; }

	private:
		GLuint m_positionBuffer{ 0 };
		GLuint m_normalBuffer{ 0 };

		int m_nVertices{ 0 };
	};


	class TextureBuffers {
	public:
		void create();
		void bind(GLuint tex);

		unsigned getNumVertices() { return m_nVertices; }

	private:
		int m_nVertices{ 0 };

		GLuint m_positionBuffer{ 0 };
		GLuint m_uvBuffer{ 0 };
		GLuint m_texBuffer{ 0 };
	};


	class GraphNodeBuffers {
	public:
		void create(MemoryContainer<float4>& nodePositions);
		void bind();

		unsigned getNumNodes() { return m_nNodes; }
		unsigned getNumVertices() { return m_nVertices; }
		unsigned getNumIndices() { return m_nIndices; }

	private:
		int m_nNodes{ 0 };
		int m_nVertices{ 0 };
		int m_nIndices{ 0 };

		GLuint m_spherePositionBuffer{ 0 };
		GLuint m_sphereNormalBuffer{ 0 };
		GLuint m_sphereEboBuffer{ 0 };
		GLuint m_nodeTransform0Buffer{ 0 };
		GLuint m_nodeTransform1Buffer{ 0 };
		GLuint m_nodeTransform2Buffer{ 0 };
		GLuint m_nodeColorBuffer{ 0 };
	};


	class GraphEdgeBuffers {
	public:
		void create(MemoryContainer<float4>& nodePositions, Grid2<int>& nodeEdges);
		void bind();

		unsigned getNumEdges() { return m_nEdges; }
		unsigned getNumVertices() { return m_nVertices; }
		unsigned getNumIndices() { return m_nIndices; }

	private:
		int m_nEdges{ 0 };
		int m_nVertices{ 0 };
		int m_nIndices{ 0 };

		GLuint m_linePositionBuffer{ 0 };
		GLuint m_lineNormalBuffer{ 0 };
		GLuint m_lineEboBuffer{ 0 };
		GLuint m_edgeTransform0Buffer{ 0 };
		GLuint m_edgeTransform1Buffer{ 0 };
		GLuint m_edgeTransform2Buffer{ 0 };
		GLuint m_edgeColorBuffer{ 0 };
	};


	class CorrespondenceBuffers {
	public:
		void create(MemoryContainer<float4>& sourcePositions, MemoryContainer<float4>& targetPositions);
		void bind();

		unsigned getNumCorrespondences() { return m_nCorrespondences; }
		unsigned getNumVertices() { return m_nVertices; }
		unsigned getNumIndices() { return m_nIndices; }

	private:
		int m_nCorrespondences{ 0 };
		int m_nVertices{ 0 };
		int m_nIndices{ 0 };

		GLuint m_linePositionBuffer{ 0 };
		GLuint m_lineNormalBuffer{ 0 };
		GLuint m_lineEboBuffer{ 0 };
		GLuint m_correspondenceTransform0Buffer{ 0 };
		GLuint m_correspondenceTransform1Buffer{ 0 };
		GLuint m_correspondenceTransform2Buffer{ 0 };
		GLuint m_correspondenceColorBuffer{ 0 };
	};


	class PointcloudBuffers {
	public:
		void create(MemoryContainer<float4>& points);
		void bind();

		unsigned getNumPoints() { return m_nPoints; }
		unsigned getNumVertices() { return m_nVertices; }
		unsigned getNumIndices() { return m_nIndices; }

	private:
		int m_nPoints{ 0 };
		int m_nVertices{ 0 };
		int m_nIndices{ 0 };

		GLuint m_boxPositionBuffer{ 0 };
		GLuint m_boxNormalBuffer{ 0 };
		GLuint m_boxEboBuffer{ 0 };
		GLuint m_pointTransform0Buffer{ 0 };
		GLuint m_pointTransform1Buffer{ 0 };
		GLuint m_pointTransform2Buffer{ 0 };
		GLuint m_pointColorBuffer{ 0 };
	};


	class ShaderPhong {
	public:
		void compile(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
		void use();
		void use(const Mat4f& viewMatrix, const Mat4f& viewProjMatrix, const Vec3f& eye, const Vec3f& lightDirection);

	private:
		GLuint m_program{ 0 };
		GLint m_viewMatrixLocation, m_viewProjMatrixLocation;
		GLint m_ambientLocation, m_diffuseLocation, m_specularLocation, m_shinyLocation;
		GLint m_lightDirLocation, m_eyeLocation;

		Vec3f m_ambient{ 0.4f, 0.4f, 0.4f };
		Vec3f m_diffuse{ 1.0f, 1.0f, 1.0f };
		Vec3f m_specular{ 1.0f, 1.0f, 1.0f };
		float m_shiny = 0.f;//10.f;
	};


	class ShaderTexture {
	public:
		void compile(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
		void use();

	private:
		GLuint m_program{ 0 };
		GLint m_samplerLocation{ 0 };
	};


	class ShaderPoints {
	public:
		void compile(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
		void use(const Mat4f& viewMatrix, const Mat4f& viewProjMatrix, const Mat4f& colorIntrinsicsMatrix, const Mat4f& colorExtrinsicsMatrix, int colorWidth, int colorHeight);

	private:
		GLuint m_program{ 0 };
		GLint m_viewMatrixLocation{ 0 };
		GLint m_viewProjMatrixLocation{ 0 };
		GLint m_colorIntrinsicsLocation{ 0 };
		GLint m_colorExtrinsicsLocation{ 0 };
		GLint m_colorWidthLocation{ 0 };
		GLint m_colorHeightLocation{ 0 };
		GLint m_samplerLocation{ 0 };
	};

} // namespace heatmap_fusion

#endif // !VISUALIZATIONHELPER_H

#include "VisualizationHelper.h"

#include "GLShader.h"
#include "TriMesh.h"
#include "MeshProcessingCPU.h"
#include "Utils.h"

namespace heatmap_fusion {

	void MeshBuffers::create(MemoryContainer<float4>& positions, MemoryContainer<float4>& normals) {
		if (m_positionBuffer) glDeleteBuffers(1, &m_positionBuffer);
		if (m_normalBuffer) glDeleteBuffers(1, &m_normalBuffer);

		positions.updateHostIfNeeded();
		normals.updateHostIfNeeded();

		m_nVertices = positions.getSize();

		glGenBuffers(1, &m_positionBuffer);
		glGenBuffers(1, &m_normalBuffer);

		glBindBuffer(GL_ARRAY_BUFFER, m_positionBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, positions.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_normalBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, normals.h(), GL_STATIC_DRAW);
	}

	void MeshBuffers::bind(int attributeOffset) {
		glBindBuffer(GL_ARRAY_BUFFER, m_positionBuffer);
		glVertexAttribPointer(attributeOffset + 0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(attributeOffset + 0);

		glBindBuffer(GL_ARRAY_BUFFER, m_normalBuffer);
		glVertexAttribPointer(attributeOffset + 1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(attributeOffset + 1);
	}

	void TextureBuffers::create() {
		if (m_positionBuffer) glDeleteBuffers(1, &m_positionBuffer);
		if (m_uvBuffer) glDeleteBuffers(1, &m_uvBuffer);

		glGenBuffers(1, &m_positionBuffer);
		glGenBuffers(1, &m_uvBuffer);

		float positions[8] = { -1, 1, -1, -1, 1, -1, 1, 1 };
		float uvs[8] = { 0, 1, 0, 0, 1, 0, 1, 1 };

		m_nVertices = 4;

		// Positions.
		glBindBuffer(GL_ARRAY_BUFFER, m_positionBuffer);
		glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(float) * m_nVertices, positions, GL_STATIC_DRAW);

		// UVs.
		glBindBuffer(GL_ARRAY_BUFFER, m_uvBuffer);
		glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(float) * m_nVertices, uvs, GL_STATIC_DRAW);
	}

	void TextureBuffers::bind(GLuint tex) {
		// Positions.
		glBindBuffer(GL_ARRAY_BUFFER, m_positionBuffer);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		// UVs.
		glBindBuffer(GL_ARRAY_BUFFER, m_uvBuffer);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		// Texture.
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	}

	void GraphNodeBuffers::create(MemoryContainer<float4>& nodePositions) {
		if (m_spherePositionBuffer) glDeleteBuffers(1, &m_spherePositionBuffer);
		if (m_sphereNormalBuffer) glDeleteBuffers(1, &m_sphereNormalBuffer);
		if (m_sphereEboBuffer) glDeleteBuffers(1, &m_sphereEboBuffer);
		if (m_nodeTransform0Buffer) glDeleteBuffers(1, &m_nodeTransform0Buffer);
		if (m_nodeTransform1Buffer) glDeleteBuffers(1, &m_nodeTransform1Buffer);
		if (m_nodeTransform2Buffer) glDeleteBuffers(1, &m_nodeTransform2Buffer);
		if (m_nodeColorBuffer) glDeleteBuffers(1, &m_nodeColorBuffer);

		float radius = 0.015f;
		TriMesh triMesh = mesh_proc::generateSphere(radius);

		triMesh.positions.updateHostIfNeeded();
		triMesh.normals.updateHostIfNeeded();

		m_nVertices = triMesh.positions.getSize();
		m_nIndices = triMesh.faceIndices.getSize();

		glGenBuffers(1, &m_spherePositionBuffer);
		glGenBuffers(1, &m_sphereNormalBuffer);
		glGenBuffers(1, &m_sphereEboBuffer);
		glGenBuffers(1, &m_nodeTransform0Buffer);
		glGenBuffers(1, &m_nodeTransform1Buffer);
		glGenBuffers(1, &m_nodeTransform2Buffer);
		glGenBuffers(1, &m_nodeColorBuffer);

		m_nNodes = nodePositions.getSize();
		nodePositions.updateHostIfNeeded();

		MemoryContainer<float4> nodeTransforms0, nodeTransforms1, nodeTransforms2;
		nodeTransforms0.allocate(m_nNodes, true, false);
		nodeTransforms1.allocate(m_nNodes, true, false);
		nodeTransforms2.allocate(m_nNodes, true, false);

		for (int i = 0; i < m_nNodes; i++) {
			Mat4f transform = Mat4f::translation(Vec3f(nodePositions.h(i)));;
			nodeTransforms0.h(i) = transform.xrow().r();
			nodeTransforms1.h(i) = transform.yrow().r();
			nodeTransforms2.h(i) = transform.zrow().r();
		}

		MemoryContainer<float4> nodeColors;
		nodeColors.allocate(m_nNodes, true, false);
		for (int i = 0; i < m_nNodes; i++) {
			nodeColors.h(i) = Vec4f(0.f, 1.f, 0.f, 1.f).r();
		}

		glBindBuffer(GL_ARRAY_BUFFER, m_spherePositionBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, triMesh.positions.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_sphereNormalBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, triMesh.normals.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_sphereEboBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * m_nIndices, triMesh.faceIndices.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_nodeTransform0Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nNodes, nodeTransforms0.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_nodeTransform1Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nNodes, nodeTransforms1.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_nodeTransform2Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nNodes, nodeTransforms2.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_nodeColorBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nNodes, nodeColors.h(), GL_STATIC_DRAW);
	}

	void GraphNodeBuffers::bind() {
		glBindBuffer(GL_ARRAY_BUFFER, m_spherePositionBuffer);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, m_sphereNormalBuffer);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, m_nodeTransform0Buffer);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);
		glVertexAttribDivisor(2, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_nodeTransform1Buffer);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(3);
		glVertexAttribDivisor(3, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_nodeTransform2Buffer);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(4);
		glVertexAttribDivisor(4, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_nodeColorBuffer);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(5);
		glVertexAttribDivisor(5, 1);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_sphereEboBuffer);
	}

	void GraphEdgeBuffers::create(MemoryContainer<float4>& nodePositions, Grid2<int>& nodeEdges) {
		if (m_linePositionBuffer) glDeleteBuffers(1, &m_linePositionBuffer);
		if (m_lineNormalBuffer) glDeleteBuffers(1, &m_lineNormalBuffer);
		if (m_lineEboBuffer) glDeleteBuffers(1, &m_lineEboBuffer);
		if (m_edgeTransform0Buffer) glDeleteBuffers(1, &m_edgeTransform0Buffer);
		if (m_edgeTransform1Buffer) glDeleteBuffers(1, &m_edgeTransform1Buffer);
		if (m_edgeTransform2Buffer) glDeleteBuffers(1, &m_edgeTransform2Buffer);
		if (m_edgeColorBuffer) glDeleteBuffers(1, &m_edgeColorBuffer);

		float thickness = 1.f; // Will be scaled later depending on edge length.
		float radius = 0.005f;
		TriMesh triMesh = mesh_proc::generateCylinder(radius, thickness);

		triMesh.positions.updateHostIfNeeded();
		triMesh.normals.updateHostIfNeeded();

		m_nVertices = triMesh.positions.getSize();
		m_nIndices = triMesh.faceIndices.getSize();

		glGenBuffers(1, &m_linePositionBuffer);
		glGenBuffers(1, &m_lineNormalBuffer);
		glGenBuffers(1, &m_lineEboBuffer);
		glGenBuffers(1, &m_edgeTransform0Buffer);
		glGenBuffers(1, &m_edgeTransform1Buffer);
		glGenBuffers(1, &m_edgeTransform2Buffer);
		glGenBuffers(1, &m_edgeColorBuffer);

		int nNodes = nodePositions.getSize();
		m_nEdges = nNodes * nodeEdges.getDimX();
		nodePositions.updateHostIfNeeded();
		nodeEdges.getContainer().updateHostIfNeeded();

		MemoryContainer<float4> edgeTransforms0, edgeTransforms1, edgeTransforms2;
		edgeTransforms0.allocate(m_nEdges, true, false);
		edgeTransforms1.allocate(m_nEdges, true, false);
		edgeTransforms2.allocate(m_nEdges, true, false);

		Grid2Interface<int, MemoryTypeCPU> iNodeEdges(nodeEdges);

		for (int nodeId = 0; nodeId < nodeEdges.getDimY(); nodeId++) {
			Vec3f p0 = nodePositions.h(nodeId);

			for (int edgeIdx = 0; edgeIdx < nodeEdges.getDimX(); edgeIdx++) {
				Vec3f p1 = nodePositions.h(iNodeEdges(edgeIdx, nodeId));
				Mat4f transform = Mat4f::translation(Vec3f(p0)) * utils::faceTransform(Vec3f(0, 0, 1), p1 - p0) * Mat4f::scale(1.0, 1.0, (p1 - p0).length());

				int edgeId = nodeId * nodeEdges.getDimX() + edgeIdx;
				edgeTransforms0.h(edgeId) = transform.xrow().r();
				edgeTransforms1.h(edgeId) = transform.yrow().r();
				edgeTransforms2.h(edgeId) = transform.zrow().r();
			}
		}

		MemoryContainer<float4> edgeColors;
		edgeColors.allocate(m_nEdges, true, false);
		for (int i = 0; i < m_nEdges; i++) {
			edgeColors.h(i) = Vec4f(1.f, 0.f, 0.f, 1.f).r();
		}

		glBindBuffer(GL_ARRAY_BUFFER, m_linePositionBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, triMesh.positions.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_lineNormalBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, triMesh.normals.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_lineEboBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * m_nIndices, triMesh.faceIndices.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_edgeTransform0Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nEdges, edgeTransforms0.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_edgeTransform1Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nEdges, edgeTransforms1.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_edgeTransform2Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nEdges, edgeTransforms2.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_edgeColorBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nEdges, edgeColors.h(), GL_STATIC_DRAW);
	}

	void GraphEdgeBuffers::bind() {
		glBindBuffer(GL_ARRAY_BUFFER, m_linePositionBuffer);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, m_lineNormalBuffer);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, m_edgeTransform0Buffer);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);
		glVertexAttribDivisor(2, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_edgeTransform1Buffer);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(3);
		glVertexAttribDivisor(3, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_edgeTransform2Buffer);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(4);
		glVertexAttribDivisor(4, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_edgeColorBuffer);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(5);
		glVertexAttribDivisor(5, 1);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_lineEboBuffer);
	}

	void CorrespondenceBuffers::create(MemoryContainer<float4>& sourcePositions, MemoryContainer<float4>& targetPositions) {
		if (m_linePositionBuffer) glDeleteBuffers(1, &m_linePositionBuffer);
		if (m_lineNormalBuffer) glDeleteBuffers(1, &m_lineNormalBuffer);
		if (m_lineEboBuffer) glDeleteBuffers(1, &m_lineEboBuffer);
		if (m_correspondenceTransform0Buffer) glDeleteBuffers(1, &m_correspondenceTransform0Buffer);
		if (m_correspondenceTransform1Buffer) glDeleteBuffers(1, &m_correspondenceTransform1Buffer);
		if (m_correspondenceTransform2Buffer) glDeleteBuffers(1, &m_correspondenceTransform2Buffer);
		if (m_correspondenceColorBuffer) glDeleteBuffers(1, &m_correspondenceColorBuffer);

		float thickness = 1.f; // Will be scaled later depending on edge length.
		float radius = 0.002f;
		TriMesh triMesh = mesh_proc::generateCylinder(radius, thickness);

		triMesh.positions.updateHostIfNeeded();
		triMesh.normals.updateHostIfNeeded();

		m_nVertices = triMesh.positions.getSize();
		m_nIndices = triMesh.faceIndices.getSize();

		glGenBuffers(1, &m_linePositionBuffer);
		glGenBuffers(1, &m_lineNormalBuffer);
		glGenBuffers(1, &m_lineEboBuffer);
		glGenBuffers(1, &m_correspondenceTransform0Buffer);
		glGenBuffers(1, &m_correspondenceTransform1Buffer);
		glGenBuffers(1, &m_correspondenceTransform2Buffer);
		glGenBuffers(1, &m_correspondenceColorBuffer);

		m_nCorrespondences = sourcePositions.getSize();
		runtime_assert(targetPositions.getSize() == m_nCorrespondences, "There should be equal number of source and target positions.");
		sourcePositions.updateHostIfNeeded();
		targetPositions.updateHostIfNeeded();

		MemoryContainer<float4> correspondenceTransforms0, correspondenceTransforms1, correspondenceTransforms2;
		correspondenceTransforms0.allocate(m_nCorrespondences, true, false);
		correspondenceTransforms1.allocate(m_nCorrespondences, true, false);
		correspondenceTransforms2.allocate(m_nCorrespondences, true, false);

		for (int correspondenceId = 0; correspondenceId < m_nCorrespondences; correspondenceId++) {
			Vec3f p0 = sourcePositions.h(correspondenceId);
			Vec3f p1 = targetPositions.h(correspondenceId);
			
			Mat4f transform = Mat4f::translation(Vec3f(p0)) * utils::faceTransform(Vec3f(0, 0, 1), p1 - p0) * Mat4f::scale(1.0, 1.0, (p1 - p0).length());

			correspondenceTransforms0.h(correspondenceId) = transform.xrow().r();
			correspondenceTransforms1.h(correspondenceId) = transform.yrow().r();
			correspondenceTransforms2.h(correspondenceId) = transform.zrow().r();
		}

		MemoryContainer<float4> correspondenceColors;
		correspondenceColors.allocate(m_nCorrespondences, true, false);
		for (int i = 0; i < m_nCorrespondences; i++) {
			correspondenceColors.h(i) = Vec4f(1.f, 0.f, 0.f, 1.f).r();
		}

		glBindBuffer(GL_ARRAY_BUFFER, m_linePositionBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, triMesh.positions.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_lineNormalBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, triMesh.normals.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_lineEboBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * m_nIndices, triMesh.faceIndices.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_correspondenceTransform0Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nCorrespondences, correspondenceTransforms0.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_correspondenceTransform1Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nCorrespondences, correspondenceTransforms1.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_correspondenceTransform2Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nCorrespondences, correspondenceTransforms2.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_correspondenceColorBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nCorrespondences, correspondenceColors.h(), GL_STATIC_DRAW);
	}

	void CorrespondenceBuffers::bind() {
		glBindBuffer(GL_ARRAY_BUFFER, m_linePositionBuffer);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, m_lineNormalBuffer);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, m_correspondenceTransform0Buffer);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);
		glVertexAttribDivisor(2, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_correspondenceTransform1Buffer);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(3);
		glVertexAttribDivisor(3, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_correspondenceTransform2Buffer);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(4);
		glVertexAttribDivisor(4, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_correspondenceColorBuffer);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(5);
		glVertexAttribDivisor(5, 1);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_lineEboBuffer);
	}

	void PointcloudBuffers::create(MemoryContainer<float4>& points) {
		if (m_boxPositionBuffer) glDeleteBuffers(1, &m_boxPositionBuffer);
		if (m_boxNormalBuffer) glDeleteBuffers(1, &m_boxNormalBuffer);
		if (m_boxEboBuffer) glDeleteBuffers(1, &m_boxEboBuffer);
		if (m_pointTransform0Buffer) glDeleteBuffers(1, &m_pointTransform0Buffer);
		if (m_pointTransform1Buffer) glDeleteBuffers(1, &m_pointTransform1Buffer);
		if (m_pointTransform2Buffer) glDeleteBuffers(1, &m_pointTransform2Buffer);
		if (m_pointColorBuffer) glDeleteBuffers(1, &m_pointColorBuffer);

		float radius = 0.005f;
		TriMesh triMesh = mesh_proc::generateBox(radius);

		triMesh.positions.updateHostIfNeeded();
		triMesh.normals.updateHostIfNeeded();

		m_nVertices = triMesh.positions.getSize();
		m_nIndices = triMesh.faceIndices.getSize();

		glGenBuffers(1, &m_boxPositionBuffer);
		glGenBuffers(1, &m_boxNormalBuffer);
		glGenBuffers(1, &m_boxEboBuffer);
		glGenBuffers(1, &m_pointTransform0Buffer);
		glGenBuffers(1, &m_pointTransform1Buffer);
		glGenBuffers(1, &m_pointTransform2Buffer);
		glGenBuffers(1, &m_pointColorBuffer);

		m_nPoints = points.getSize();
		points.updateHostIfNeeded();

		MemoryContainer<float4> pointTransforms0, pointTransforms1, pointTransforms2;
		pointTransforms0.allocate(m_nPoints, true, false);
		pointTransforms1.allocate(m_nPoints, true, false);
		pointTransforms2.allocate(m_nPoints, true, false);

		for (int i = 0; i < m_nPoints; i++) {
			Mat4f transform = Mat4f::translation(Vec3f(points.h(i)));;
			pointTransforms0.h(i) = transform.xrow().r();
			pointTransforms1.h(i) = transform.yrow().r();
			pointTransforms2.h(i) = transform.zrow().r();
		}

		MemoryContainer<float4> pointColors;
		pointColors.allocate(m_nPoints, true, false);
		for (int i = 0; i < m_nPoints; i++) {
			pointColors.h(i) = Vec4f(1.f, 0.f, 0.f, 1.f).r();
		}

		glBindBuffer(GL_ARRAY_BUFFER, m_boxPositionBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, triMesh.positions.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_boxNormalBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nVertices, triMesh.normals.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_boxEboBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * m_nIndices, triMesh.faceIndices.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_pointTransform0Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nPoints, pointTransforms0.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_pointTransform1Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nPoints, pointTransforms1.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_pointTransform2Buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nPoints, pointTransforms2.h(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, m_pointColorBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*m_nPoints, pointColors.h(), GL_STATIC_DRAW);
	}

	void PointcloudBuffers::bind() {
		glBindBuffer(GL_ARRAY_BUFFER, m_boxPositionBuffer);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, m_boxNormalBuffer);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, m_pointTransform0Buffer);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);
		glVertexAttribDivisor(2, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_pointTransform1Buffer);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(3);
		glVertexAttribDivisor(3, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_pointTransform2Buffer);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(4);
		glVertexAttribDivisor(4, 1);

		glBindBuffer(GL_ARRAY_BUFFER, m_pointColorBuffer);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(5);
		glVertexAttribDivisor(5, 1);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_boxEboBuffer);
	}

	void ShaderPhong::compile(const std::string& vertexShaderPath, const std::string& fragmentShaderPath) {
		m_program = LoadShader(vertexShaderPath.c_str(), fragmentShaderPath.c_str());

		// Link phong shading parameters.
		glUseProgram(m_program);

		m_viewMatrixLocation = glGetUniformLocation(m_program, "g_viewMatrix");
		m_viewProjMatrixLocation = glGetUniformLocation(m_program, "g_viewProjMatrix");
		m_ambientLocation = glGetUniformLocation(m_program, "g_ambient");
		m_diffuseLocation = glGetUniformLocation(m_program, "g_diffuse");
		m_specularLocation = glGetUniformLocation(m_program, "g_specular");
		m_shinyLocation = glGetUniformLocation(m_program, "g_shiny");
		m_lightDirLocation = glGetUniformLocation(m_program, "g_lightDir");
		m_eyeLocation = glGetUniformLocation(m_program, "g_eye");
	}

	void ShaderPhong::use() {
		glUseProgram(m_program);
	}

	void ShaderPhong::use(const Mat4f& viewMatrix, const Mat4f& viewProjMatrix, const Vec3f& eye, const Vec3f& lightDirection) {
		glUseProgram(m_program);

		glUniformMatrix4fv(m_viewMatrixLocation, 1, GL_TRUE, (const GLfloat*)viewMatrix.getData());
		glUniformMatrix4fv(m_viewProjMatrixLocation, 1, GL_TRUE, (const GLfloat*)viewProjMatrix.getData());
		glUniform3f(m_ambientLocation, m_ambient.x(), m_ambient.y(), m_ambient.z());
		glUniform3f(m_diffuseLocation, m_diffuse.x(), m_diffuse.y(), m_diffuse.z());
		glUniform3f(m_specularLocation, m_specular.x(), m_specular.y(), m_specular.z());
		glUniform1f(m_shinyLocation, m_shiny);
		glUniform3f(m_lightDirLocation, lightDirection.x(), lightDirection.y(), lightDirection.z());
		glUniform3f(m_eyeLocation, eye.x(), eye.y(), eye.z());
	}

	void ShaderTexture::compile(const std::string& vertexShaderPath, const std::string& fragmentShaderPath) {
		m_program = LoadShader(vertexShaderPath.c_str(), fragmentShaderPath.c_str());

		// Link phong shading parameters.
		glUseProgram(m_program);

		m_samplerLocation = glGetUniformLocation(m_program, "g_sampler");
	}

	void ShaderTexture::use() {
		glUseProgram(m_program);

		glUniform1i(m_samplerLocation, 0);
	}

	void ShaderPoints::compile(const std::string& vertexShaderPath, const std::string& fragmentShaderPath) {
		m_program = LoadShader(vertexShaderPath.c_str(), fragmentShaderPath.c_str());

		// Link parameters.
		glUseProgram(m_program);

		m_viewMatrixLocation = glGetUniformLocation(m_program, "g_viewMatrix");
		m_viewProjMatrixLocation = glGetUniformLocation(m_program, "g_viewProjMatrix");
		m_colorIntrinsicsLocation = glGetUniformLocation(m_program, "g_colorIntrinsics");
		m_colorExtrinsicsLocation = glGetUniformLocation(m_program, "g_colorExtrinsics");
		m_colorWidthLocation = glGetUniformLocation(m_program, "g_colorWidth");
		m_colorHeightLocation = glGetUniformLocation(m_program, "g_colorHeight");
		m_samplerLocation = glGetUniformLocation(m_program, "g_sampler");
	}

	void ShaderPoints::use(const Mat4f& viewMatrix, const Mat4f& viewProjMatrix, const Mat4f& colorIntrinsicsMatrix, const Mat4f& colorExtrinsicsMatrix, int colorWidth, int colorHeight) {
		glUseProgram(m_program);

		glUniformMatrix4fv(m_viewMatrixLocation, 1, GL_TRUE, (const GLfloat*)viewMatrix.getData());
		glUniformMatrix4fv(m_viewProjMatrixLocation, 1, GL_TRUE, (const GLfloat*)viewProjMatrix.getData());
		glUniformMatrix4fv(m_colorIntrinsicsLocation, 1, GL_TRUE, (const GLfloat*)colorIntrinsicsMatrix.getData());
		glUniformMatrix4fv(m_colorExtrinsicsLocation, 1, GL_TRUE, (const GLfloat*)colorExtrinsicsMatrix.getData());
		glUniform1i(m_colorWidthLocation, colorWidth);
		glUniform1i(m_colorHeightLocation, colorHeight);
		glUniform1i(m_samplerLocation, 0);
	}

	

} // namespace heatmap_fusion
#include "MeshProcessingCPU.h"

#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

using namespace matrix_lib;

namespace heatmap_fusion {
	namespace mesh_proc {

		void computeMeshNormals(MemoryContainer<float4>& positions, MemoryContainer<unsigned>& faceIndices, MemoryContainer<float4>& normals) {
			int nVertices = positions.getSize();
			int nFaces = faceIndices.getSize() / 3;
			if (nVertices == 0 || nFaces == 0) return;

			positions.updateHostIfNeeded();
			faceIndices.updateHostIfNeeded();

			normals.clear();
			normals.allocate(nVertices, true, false);

			for (int i = 0; i < nVertices; i++) {
				normals.h(i) = make_float4(0, 0, 0, 0);
			}

			for (int i = 0; i < nFaces; i++) {
				Vec3i face(faceIndices.h(3 * i + 0), faceIndices.h(3 * i + 1) , faceIndices.h(3 * i + 2));

				Vec3f v0(positions.h(face.x()));
				Vec3f v1(positions.h(face.y()));
				Vec3f v2(positions.h(face.z()));

				Vec3f faceNormal = (v1 - v0) ^ (v2 - v0);

				normals.h(face.x()) = (Vec3f(normals.h(face.x())) + faceNormal).r();
				normals.h(face.y()) = (Vec3f(normals.h(face.y())) + faceNormal).r();
				normals.h(face.z()) = (Vec3f(normals.h(face.z())) + faceNormal).r();
			}

			for (int i = 0; i < nVertices; i++) {
				Vec3f n(normals.h(i));
				n.normalizeIfNonzero();
				normals.h(i) = n.r();
			}

			normals.setUpdated(true, false);
		}

		VertexMesh convertMesh(TriMesh& triMesh) {
			int nVertices = triMesh.positions.getSize();
			int nFaceIndices = triMesh.faceIndices.getSize();
			if (nVertices == 0 || nFaceIndices == 0) return VertexMesh();
			
			bool bHasNormals = triMesh.normals.getSize() > 0;
			bool bHasColors = triMesh.colors.getSize() > 0;

			VertexMesh vertexMesh;
			vertexMesh.positions.allocate(nFaceIndices);
			triMesh.positions.updateHostIfNeeded();
			triMesh.faceIndices.updateHostIfNeeded();
			
			if (bHasNormals) {
				vertexMesh.normals.allocate(nFaceIndices);
				triMesh.normals.updateHostIfNeeded();
			}
			if (bHasColors) {
				vertexMesh.colors.allocate(nFaceIndices);
				triMesh.colors.updateHostIfNeeded();
			}

			for (int i = 0; i < nFaceIndices; i++) {
				int vertexIdx = triMesh.faceIndices.h(i);

				vertexMesh.positions.h(i) = triMesh.positions.h(vertexIdx);

				if (bHasNormals) {
					vertexMesh.normals.h(i) = triMesh.normals.h(vertexIdx);
				}

				if (bHasColors) {
					vertexMesh.colors.h(i) = triMesh.colors.h(vertexIdx);
				}
			}

			vertexMesh.positions.setUpdated(true, false);
			if (bHasNormals) vertexMesh.normals.setUpdated(true, false);
			if (bHasColors) vertexMesh.colors.setUpdated(true, false);

			return vertexMesh;
		}

		TriMesh generateSphere(float radius, int stacks, int slices, const Vec4f& color) {
			vector<Vec3f> positions, normals;
			vector<Vec4f> colors;
			vector<Vec3i> faceIndices;

			Vec3f center(0.f, 0.f, 0.f);

			const float thetaDivisor = 1.0f / stacks * math_proc::PIf;
			const float phiDivisor = 1.0f / slices * 2.0f * math_proc::PIf;
			for (int t = 0; t < stacks; t++) { // stacks increment elevation (theta)
				float theta1 = t * thetaDivisor;
				float theta2 = (t + 1) * thetaDivisor;

				for (int p = 0; p < slices; p++) { // slices increment azimuth (phi)
					float phi1 = p * phiDivisor;
					float phi2 = (p + 1) * phiDivisor;

					const auto sph2xyz = [&](float r, float theta, float phi) {
						const float sinTheta = sinf(theta), sinPhi = sinf(phi), cosTheta = cosf(theta), cosPhi = cosf(phi);
						return Vec3f(r * sinTheta * cosPhi, r * sinTheta * sinPhi, r * cosTheta);
					};

					// phi2   phi1
					//  |      |
					//  2------1 -- theta1
					//  |\ _   |
					//  |    \ |
					//  3------4 -- theta2
					//  
					// Points
					const Vec3<float>
						r1 = sph2xyz(radius, theta1, phi1),
						r2 = sph2xyz(radius, theta1, phi2),
						r3 = sph2xyz(radius, theta2, phi2),
						r4 = sph2xyz(radius, theta2, phi1);
					positions.push_back(r1 + center);
					positions.push_back(r2 + center);
					positions.push_back(r3 + center);
					positions.push_back(r4 + center);

					// Colors
					for (int i = 0; i < 4; i++) {
						colors.push_back(color);
					}

					// Normals
					normals.push_back(r1.getNormalized());
					normals.push_back(r2.getNormalized());
					normals.push_back(r3.getNormalized());
					normals.push_back(r4.getNormalized());

					const int baseIdx = static_cast<int>(t * slices * 4 + p * 4);

					// Indices
					Vec3i indices;
					if (t == 0) {  // top cap -- t1p1, t2p2, t2p1
						indices[0] = baseIdx + 0;
						indices[1] = baseIdx + 3;
						indices[2] = baseIdx + 2;
						faceIndices.push_back(indices);
					}
					else if (t + 1 == stacks) {  // bottom cap -- t2p2, t1p1, t1p2
						indices[0] = baseIdx + 2;
						indices[1] = baseIdx + 1;
						indices[2] = baseIdx + 0;
						faceIndices.push_back(indices);
					}
					else {  // regular piece
						indices[0] = baseIdx + 0;
						indices[1] = baseIdx + 3;
						indices[2] = baseIdx + 1;
						faceIndices.push_back(indices);
						
						indices[0] = baseIdx + 1;
						indices[1] = baseIdx + 3;
						indices[2] = baseIdx + 2;
						faceIndices.push_back(indices);
					}
				}
			}

			int nVertices = positions.size();
			int nFaces = faceIndices.size();

			TriMesh mesh;
			mesh.positions.allocate(nVertices, true, false);
			mesh.normals.allocate(nVertices, true, false);
			mesh.colors.allocate(nVertices, true, false);
			mesh.faceIndices.allocate(3 * nFaces, true, false);

			for (int i = 0; i < nVertices; i++) {
				mesh.positions.h(i) = positions[i].r();
				mesh.normals.h(i) = normals[i].r();
				mesh.colors.h(i) = colors[i].r();
			}
			
			for (int i = 0; i < nFaces; i++) {
				mesh.faceIndices.h(3 * i + 0) = faceIndices[i].x();
				mesh.faceIndices.h(3 * i + 1) = faceIndices[i].y();
				mesh.faceIndices.h(3 * i + 2) = faceIndices[i].z();
			}

			mesh.positions.setUpdated(true, false);
			mesh.normals.setUpdated(true, false);
			mesh.colors.setUpdated(true, false);
			mesh.faceIndices.setUpdated(true, false);

			return mesh;
		}

		TriMesh generateCylinder(float radius, float height, unsigned stacks, unsigned slices, const Vec4f& color) {
			vector<Vec3f> positions((stacks + 1) * slices);
			vector<Vec4f> colors((stacks + 1) * slices);
			vector<unsigned int> indices(stacks * slices * 6);

			unsigned vIndex = 0;
			for (unsigned i = 0; i <= stacks; i++) {
				for (unsigned i2 = 0; i2 < slices; i2++) {
					float theta = float(i2) * 2.0f * math_proc::PIf / float(slices);
					positions[vIndex] = Vec3f(radius * cosf(theta), radius * sinf(theta), height * float(i) / float(stacks));
					colors[vIndex] = color;
					vIndex++;
				}
			}

			unsigned iIndex = 0;
			for (unsigned i = 0; i < stacks; i++) {
				for (unsigned i2 = 0; i2 < slices; i2++) {
					int i2p1 = (i2 + 1) % slices;

					indices[iIndex++] = (i + 1) * slices + i2;
					indices[iIndex++] = i * slices + i2;
					indices[iIndex++] = i * slices + i2p1;


					indices[iIndex++] = (i + 1) * slices + i2;
					indices[iIndex++] = i * slices + i2p1;
					indices[iIndex++] = (i + 1) * slices + i2p1;
				}
			}

			int nVertices = positions.size();
			int nFaceIndices = indices.size();

			TriMesh mesh;
			mesh.positions.allocate(nVertices, true, false);
			mesh.colors.allocate(nVertices, true, false);
			mesh.faceIndices.allocate(nFaceIndices, true, false);

			for (int i = 0; i < nVertices; i++) {
				mesh.positions.h(i) = positions[i].r();
				mesh.colors.h(i) = colors[i].r();
			}

			for (int i = 0; i < nFaceIndices; i++) {
				mesh.faceIndices.h(i) = indices[i];
			}

			mesh.positions.setUpdated(true, false);
			mesh.colors.setUpdated(true, false);
			mesh.faceIndices.setUpdated(true, false);

			computeMeshNormals(mesh.positions, mesh.faceIndices, mesh.normals);

			mesh.normals.setUpdated(true, false);

			return mesh;
		}

		TriMesh generateBox(float xDim, float yDim, float zDim, const Vec4f& color, const Vec3f& offset) {
			float cubeVData[8][3] = {
			{ 1.0f, 1.0f, 1.0f }, { -1.0f, 1.0f, 1.0f }, { -1.0f, -1.0f, 1.0f },
			{ 1.0f, -1.0f, 1.0f }, { 1.0f, 1.0f, -1.0f }, { -1.0f, 1.0f, -1.0f },
			{ -1.0f, -1.0f, -1.0f }, { 1.0f, -1.0f, -1.0f }
			};

			int cubeIData[12][3] = {
				{ 1, 2, 3 }, { 1, 3, 0 }, { 0, 3, 7 }, { 0, 7, 4 }, { 3, 2, 6 },
				{ 3, 6, 7 }, { 1, 6, 2 }, { 1, 5, 6 }, { 0, 5, 1 }, { 0, 4, 5 },
				{ 6, 5, 4 }, { 6, 4, 7 }
			};

			int cubeEData[12][2] = {
				{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
				{ 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
				{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
			};

			int nVertices = 8;
			int nFaceIndices = 12 * 3;

			float sx = 0.5f * xDim;
			float sy = 0.5f * yDim;
			float sz = 0.5f * zDim;

			TriMesh mesh;
			mesh.positions.allocate(nVertices, true, false);
			mesh.colors.allocate(nVertices, true, false);
			mesh.faceIndices.allocate(nFaceIndices, true, false);
			
			// Vertices
			for (int i = 0; i < 8; i++) {
				mesh.positions.h(i) = make_float4(sx * cubeVData[i][0] + offset.x(), sy * cubeVData[i][1] + offset.y(), sz* cubeVData[i][2] + offset.z(), 1.f);
				mesh.colors.h(i) = color.r();
			}

			// Triangles
			for (int i = 0; i < 12; i++) {
				mesh.faceIndices.h(i * 3 + 0) = cubeIData[i][0];
				mesh.faceIndices.h(i * 3 + 1) = cubeIData[i][1];
				mesh.faceIndices.h(i * 3 + 2) = cubeIData[i][2];
			}

			mesh.positions.setUpdated(true, false);
			mesh.colors.setUpdated(true, false);
			mesh.faceIndices.setUpdated(true, false);

			computeMeshNormals(mesh.positions, mesh.faceIndices, mesh.normals);

			mesh.normals.setUpdated(true, false);

			return mesh;
		}

		TriMesh generateBox(float thickness, const Vec4f& color) {
			return generateBox(thickness, thickness, thickness, color);
		}

		TriMesh generateBox(const BoundingBox3f& boundingBox, const Vec4f& color) {
			Vec3f extent = boundingBox.getExtent();
			Vec3f center = boundingBox.getCenter();
			
			return generateBox(extent.x(), extent.y(), extent.z(), color, center);
		}

	} // namespace mesh_proc
} // namespace heatmap_fusion

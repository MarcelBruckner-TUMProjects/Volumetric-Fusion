#ifndef MESHPROCESSINGCPU_H
#define MESHPROCESSINGCPU_H

#include <vector_types.h>
#include <vector_functions.h>
#include <common_utils/memory_managment/MemoryContainer.h>
#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

#include "VertexMesh.h"
#include "TriMesh.h"
#include "Texture.h"
#include "BoundingBox.h"

using namespace common_utils;
using namespace matrix_lib;

namespace heatmap_fusion {
	namespace mesh_proc {

		void computeMeshNormals(MemoryContainer<float4>& positions, MemoryContainer<unsigned>& faceIndices, MemoryContainer<float4>& normals);
		
		VertexMesh convertMesh(TriMesh& triMesh);

		TriMesh generateSphere(float radius, int stacks = 10, int slices = 10, const Vec4f& color = Vec4f(1,1,1,1));

		TriMesh generateCylinder(float radius, float height, unsigned stacks = 2, unsigned slices = 9, const Vec4f& color = Vec4f(1, 1, 1, 1));
		
		TriMesh generateBox(float xDim, float yDim, float zDim, const Vec4f& color = Vec4f(1, 1, 1, 1), const Vec3f& offset = Vec3f(0, 0, 0));
		TriMesh generateBox(float thickness, const Vec4f& color = Vec4f(1, 1, 1, 1));
		TriMesh generateBox(const BoundingBox3f& boundingBox, const Vec4f& color = Vec4f(1, 1, 1, 1));

	} // namespace mesh_proc
} // namespace heatmap_fusion

#endif // !MESHPROCESSINGCPU_H

#ifndef VERTEXMESH_H
#define VERTEXMESH_H

#include <common_utils/memory_managment/MemoryContainer.h>
#include <vector_types.h>
#include <vector_functions.h>

namespace heatmap_fusion {

	using namespace common_utils;

	/**
	 * Each 3 consecutive vertices make up a triangle.
	 */
	class VertexMesh {
	public:
		MemoryContainer<float4> positions;
		MemoryContainer<float4> normals;
		MemoryContainer<float4> colors;
	};

} // namespace heatmap_fusion

#endif //VERTEXMESH_H
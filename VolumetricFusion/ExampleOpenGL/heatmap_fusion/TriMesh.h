#ifndef TRIMESH_H
#define TRIMESH_H

#include <common_utils/memory_managment/MemoryContainer.h>
#include <vector_types.h>
#include <vector_functions.h>

namespace heatmap_fusion {

	using namespace common_utils;

	class TriMesh {
	public:
		MemoryContainer<float4> positions;
		MemoryContainer<float4> normals;
		MemoryContainer<float4> colors;
		MemoryContainer<unsigned int> faceIndices;
	};

} // namespace heatmap_fusion

#endif //TRIMESH_H
#pragma once

namespace common_utils {
	
	enum class MemoryType{
		CPU_MEMORY,
		CUDA_MEMORY,
		CUSTOM_MEMORY
	};

	struct MemoryTypeCPU;
	struct MemoryTypeCUDA;
	struct MemoryTypeCustom;

} // namespace common_utils
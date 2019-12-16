#pragma once
#include <common_utils/meta_structures/BasicTypes.h>
#include <common_utils/memory_managment/MemoryType.h>

namespace solo {
	namespace memory_proc {

		/**
		 * Just function definition.
		 */
		template<typename Type>
		void copyMemory(Type* outputDataPtr, const Type* inputDataPtr, unsigned size, Type2Type<MemoryTypeCUDA>) {
			CUDA_SAFE_CALL(cudaMemcpy(outputDataPtr, inputDataPtr, size * sizeof(Type), cudaMemcpyDeviceToDevice));
		}

	} // namespace memory_proc
} // namespace solo
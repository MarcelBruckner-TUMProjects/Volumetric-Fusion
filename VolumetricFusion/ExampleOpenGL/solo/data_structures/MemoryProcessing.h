#pragma once
#include <common_utils/meta_structures/BasicTypes.h>
#include <common_utils/memory_managment/MemoryType.h>

namespace solo {
	namespace memory_proc {

		/**
		 * Copies memory from host to host.
		 */
		template<typename Type>
		void copyMemory(Type* outputDataPtr, const Type* inputDataPtr, unsigned size, Type2Type<MemoryTypeCPU>) {
			memcpy(outputDataPtr, inputDataPtr, size * sizeof(Type));
		}

		/**
		 * Copies memory from device to device.
		 */
		template<typename Type>
		void copyMemory(Type* outputDataPtr, const Type* inputDataPtr, unsigned size, Type2Type<MemoryTypeCUDA>);

	} // namespace memory_proc
} // namespace solo